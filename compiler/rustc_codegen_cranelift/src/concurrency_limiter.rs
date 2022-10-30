use std::sync::{Arc, Condvar, Mutex};

use rustc_session::Session;

use jobserver::HelperThread;

// FIXME don't panic when a worker thread panics

pub(super) struct ConcurrencyLimiter {
    helper_thread: Option<HelperThread>,
    state: Arc<Mutex<state::ConcurrencyLimiterState>>,
    available_token_condvar: Arc<Condvar>,
}

impl ConcurrencyLimiter {
    pub(super) fn new(sess: &Session, pending_jobs: usize) -> Self {
        let state = Arc::new(Mutex::new(state::ConcurrencyLimiterState::new(pending_jobs)));
        let available_token_condvar = Arc::new(Condvar::new());

        let state_helper = state.clone();
        let available_token_condvar_helper = available_token_condvar.clone();
        let helper_thread = sess
            .jobserver
            .clone()
            .into_helper_thread(move |token| {
                let mut state = state_helper.lock().unwrap();
                state.add_new_token(token.unwrap());
                available_token_condvar_helper.notify_one();
            })
            .unwrap();
        ConcurrencyLimiter {
            helper_thread: Some(helper_thread),
            state,
            available_token_condvar: Arc::new(Condvar::new()),
        }
    }

    pub(super) fn acquire(&mut self) -> ConcurrencyLimiterToken {
        let mut state = self.state.lock().unwrap();
        loop {
            state.assert_invariants();

            if state.try_start_job() {
                return ConcurrencyLimiterToken {
                    state: self.state.clone(),
                    available_token_condvar: self.available_token_condvar.clone(),
                };
            }

            self.helper_thread.as_mut().unwrap().request_token();
            state = self.available_token_condvar.wait(state).unwrap();
        }
    }

    pub(super) fn job_already_done(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.job_already_done();
    }
}

impl Drop for ConcurrencyLimiter {
    fn drop(&mut self) {
        //
        self.helper_thread.take();

        // Assert that all jobs have finished
        let state = Mutex::get_mut(Arc::get_mut(&mut self.state).unwrap()).unwrap();
        state.assert_done();
    }
}

#[derive(Debug)]
pub(super) struct ConcurrencyLimiterToken {
    state: Arc<Mutex<state::ConcurrencyLimiterState>>,
    available_token_condvar: Arc<Condvar>,
}

impl Drop for ConcurrencyLimiterToken {
    fn drop(&mut self) {
        let mut state = self.state.lock().unwrap();
        state.job_finished();
        self.available_token_condvar.notify_one();
    }
}

mod state {
    use jobserver::Acquired;

    #[derive(Debug)]
    pub(super) struct ConcurrencyLimiterState {
        pending_jobs: usize,
        active_jobs: usize,

        // None is used to represent the implicit token, Some to represent explicit tokens
        tokens: Vec<Option<Acquired>>,
    }

    impl ConcurrencyLimiterState {
        pub(super) fn new(pending_jobs: usize) -> Self {
            ConcurrencyLimiterState { pending_jobs, active_jobs: 0, tokens: vec![None] }
        }

        pub(super) fn assert_invariants(&self) {
            // There must be no excess active jobs
            assert!(self.active_jobs <= self.pending_jobs);

            // There may not be more active jobs than there are tokens
            assert!(self.active_jobs <= self.tokens.len());
        }

        pub(super) fn assert_done(&self) {
            assert_eq!(self.pending_jobs, 0);
            assert_eq!(self.active_jobs, 0);
        }

        pub(super) fn add_new_token(&mut self, token: Acquired) {
            self.tokens.push(Some(token));
            self.drop_excess_capacity();
        }

        pub(super) fn try_start_job(&mut self) -> bool {
            if self.active_jobs < self.tokens.len() {
                // Using existing token
                self.job_started();
                return true;
            }

            false
        }

        pub(super) fn job_started(&mut self) {
            self.assert_invariants();
            self.active_jobs += 1;
            self.drop_excess_capacity();
            self.assert_invariants();
        }

        pub(super) fn job_finished(&mut self) {
            self.assert_invariants();
            self.pending_jobs -= 1;
            self.active_jobs -= 1;
            self.assert_invariants();
            self.drop_excess_capacity();
            self.assert_invariants();
        }

        pub(super) fn job_already_done(&mut self) {
            self.assert_invariants();
            self.pending_jobs -= 1;
            self.assert_invariants();
            self.drop_excess_capacity();
            self.assert_invariants();
        }

        fn drop_excess_capacity(&mut self) {
            self.assert_invariants();

            // Drop all tokens that can never be used anymore
            self.tokens.truncate(std::cmp::max(self.pending_jobs, 1));

            // Keep some excess tokens to satisfy requests faster
            const MAX_EXTRA_CAPACITY: usize = 2;
            self.tokens.truncate(std::cmp::max(self.active_jobs + MAX_EXTRA_CAPACITY, 1));

            self.assert_invariants();
        }
    }
}
