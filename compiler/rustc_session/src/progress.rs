use std::{
    cell::RefCell,
    sync::mpsc::{Sender, TryRecvError},
    thread::ThreadId,
    time::Duration,
};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rustc_data_structures::{fx::FxHashMap, sync::IntoDynSyncSend};

use crate::Session;

thread_local! {
    static CURRENT_SPINNER: RefCell<Option<(ProgressBar, usize)>> = RefCell::new(None);
}

pub struct ProgressBars {
    sender: IntoDynSyncSend<Sender<Msg>>,
}

enum Msg {
    Pop { thread: ThreadId },
    Push { thread: ThreadId, name: &'static str },
}

impl Session {
    /// Starts up a thread that makes sure all the threads' messages are collected and processed
    /// in one central location, and thus rendered correctly.
    pub(crate) fn init_progress_bars() -> ProgressBars {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let bars = MultiProgress::new();
            let mut threads: FxHashMap<ThreadId, Vec<_>> = FxHashMap::default();
            'outer: loop {
                std::thread::sleep(Duration::from_millis(100));
                loop {
                    match receiver.try_recv() {
                        Ok(val) => match val {
                            Msg::Pop { thread } => {
                                threads.get_mut(&thread).unwrap().pop();
                            }
                            Msg::Push { thread, name } => {
                                let stack = threads.entry(thread).or_default();

                                let mut template = String::new();
                                use std::fmt::Write;
                                if !stack.is_empty() {
                                    for _ in 1..stack.len() {
                                        write!(template, " ").unwrap();
                                    }
                                    write!(template, "└").unwrap();
                                }
                                write!(template, "{{spinner}} {{msg}}").unwrap();

                                let spinner = ProgressBar::new_spinner()
                                    .with_message(name)
                                    .with_style(ProgressStyle::with_template(&template).unwrap());
                                let spinner = bars.add(spinner);
                                stack.push(spinner)
                            }
                        },
                        Err(TryRecvError::Disconnected) => break 'outer,
                        Err(TryRecvError::Empty) => break,
                    }
                }
                for thread in threads.values() {
                    for spinner in thread {
                        spinner.tick()
                    }
                }
            }
        });
        ProgressBars { sender: IntoDynSyncSend(sender) }
    }

    /// Append a new spinner to the current stack
    pub fn push_spinner(&self, bars: &ProgressBars, name: &'static str) -> impl Sized {
        let thread = std::thread::current().id();
        bars.sender.send(Msg::Push { thread, name }).unwrap();
        struct Spinner(Sender<Msg>, ThreadId);
        impl Drop for Spinner {
            fn drop(&mut self) {
                self.0.send(Msg::Pop { thread: self.1 }).unwrap();
            }
        }
        Spinner(bars.sender.0.clone(), thread)
    }
}
