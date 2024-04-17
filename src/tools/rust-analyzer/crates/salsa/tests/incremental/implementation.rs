use crate::constants;
use crate::counter::Counter;
use crate::log::Log;
use crate::memoized_dep_inputs;
use crate::memoized_inputs;
use crate::memoized_volatile;

pub(crate) trait TestContext: salsa::Database {
    fn clock(&self) -> &Counter;
    fn log(&self) -> &Log;
}

#[salsa::database(
    constants::Constants,
    memoized_dep_inputs::MemoizedDepInputs,
    memoized_inputs::MemoizedInputs,
    memoized_volatile::MemoizedVolatile
)]
#[derive(Default)]
pub(crate) struct TestContextImpl {
    storage: salsa::Storage<TestContextImpl>,
    clock: Counter,
    log: Log,
}

impl TestContextImpl {
    #[track_caller]
    pub(crate) fn assert_log(&self, expected_log: &[&str]) {
        let expected_text = &format!("{:#?}", expected_log);
        let actual_text = &format!("{:#?}", self.log().take());

        if expected_text == actual_text {
            return;
        }

        #[allow(clippy::print_stdout)]
        for diff in dissimilar::diff(expected_text, actual_text) {
            match diff {
                dissimilar::Chunk::Delete(l) => println!("-{}", l),
                dissimilar::Chunk::Equal(l) => println!(" {}", l),
                dissimilar::Chunk::Insert(r) => println!("+{}", r),
            }
        }

        panic!("incorrect log results");
    }
}

impl TestContext for TestContextImpl {
    fn clock(&self) -> &Counter {
        &self.clock
    }

    fn log(&self) -> &Log {
        &self.log
    }
}

impl salsa::Database for TestContextImpl {}
