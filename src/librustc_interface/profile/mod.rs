use log::debug;
use rustc::dep_graph::DepNode;
use rustc::session::Session;
use rustc::util::common::{ProfQDumpParams, ProfileQueriesMsg, profq_msg, profq_set_chan};
use std::sync::mpsc::{Receiver};
use std::io::{Write};
use std::time::{Duration, Instant};

pub mod trace;

/// begin a profile thread, if not already running
pub fn begin(sess: &Session) {
    use std::thread;
    use std::sync::mpsc::{channel};
    let (tx, rx) = channel();
    if profq_set_chan(sess, tx) {
        thread::spawn(move || profile_queries_thread(rx));
    }
}

/// dump files with profiling information to the given base path, and
/// wait for this dump to complete.
///
/// wraps the RPC (send/recv channel logic) of requesting a dump.
pub fn dump(sess: &Session, path: String) {
    use std::sync::mpsc::{channel};
    let (tx, rx) = channel();
    let params = ProfQDumpParams {
        path,
        ack: tx,
        // FIXME: Add another compiler flag to toggle whether this log
        // is written; false for now
        dump_profq_msg_log: true,
    };
    profq_msg(sess, ProfileQueriesMsg::Dump(params));
    let _ = rx.recv().unwrap();
}

// State for parsing recursive trace structure in separate thread, via messages
#[derive(Clone, Eq, PartialEq)]
enum ParseState {
    // No (local) parse state; may be parsing a tree, focused on a
    // sub-tree that could be anything.
    Clear,
    // Have Query information from the last message
    HaveQuery(trace::Query, Instant),
    // Have "time-begin" information from the last message (doit flag, and message)
    HaveTimeBegin(String, Instant),
    // Have "task-begin" information from the last message
    HaveTaskBegin(DepNode, Instant),
}
struct StackFrame {
    pub parse_st: ParseState,
    pub traces:   Vec<trace::Rec>,
}

fn total_duration(traces: &[trace::Rec]) -> Duration {
    Duration::new(0, 0) + traces.iter().map(|t| t.dur_total).sum()
}

// profiling thread; retains state (in local variables) and dump traces, upon request.
fn profile_queries_thread(r: Receiver<ProfileQueriesMsg>) {
    use self::trace::*;
    use std::fs::File;

    let mut profq_msgs: Vec<ProfileQueriesMsg> = vec![];
    let mut frame: StackFrame = StackFrame { parse_st: ParseState::Clear, traces: vec![] };
    let mut stack: Vec<StackFrame> = vec![];
    loop {
        let msg = r.recv();
        if let Err(_recv_err) = msg {
            // FIXME: Perhaps do something smarter than simply quitting?
            break
        };
        let msg = msg.unwrap();
        debug!("profile_queries_thread: {:?}", msg);

        // Meta-level versus _actual_ queries messages
        match msg {
            ProfileQueriesMsg::Halt => return,
            ProfileQueriesMsg::Dump(params) => {
                assert!(stack.is_empty());
                assert!(frame.parse_st == ParseState::Clear);

                // write log of all messages
                if params.dump_profq_msg_log {
                    let mut log_file =
                        File::create(format!("{}.log.txt", params.path)).unwrap();
                    for m in profq_msgs.iter() {
                        writeln!(&mut log_file, "{:?}", m).unwrap()
                    };
                }

                // write HTML file, and counts file
                let html_path = format!("{}.html", params.path);
                let mut html_file = File::create(&html_path).unwrap();

                let counts_path = format!("{}.counts.txt", params.path);
                let mut counts_file = File::create(&counts_path).unwrap();

                writeln!(html_file,
                    "<html>\n<head>\n<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\">",
                    "profile_queries.css").unwrap();
                writeln!(html_file, "<style>").unwrap();
                trace::write_style(&mut html_file);
                writeln!(html_file, "</style>\n</head>\n<body>").unwrap();
                trace::write_traces(&mut html_file, &mut counts_file, &frame.traces);
                writeln!(html_file, "</body>\n</html>").unwrap();

                let ack_path = format!("{}.ack", params.path);
                let ack_file = File::create(&ack_path).unwrap();
                drop(ack_file);

                // Tell main thread that we are done, e.g., so it can exit
                params.ack.send(()).unwrap();
            }
            // Actual query message:
            msg => {
                // Record msg in our log
                profq_msgs.push(msg.clone());
                // Respond to the message, knowing that we've already handled Halt and Dump, above.
                match (frame.parse_st.clone(), msg) {
                    (_, ProfileQueriesMsg::Halt) | (_, ProfileQueriesMsg::Dump(_)) => {
                        unreachable!();
                    },
                    // Parse State: Clear
                    (ParseState::Clear,
                     ProfileQueriesMsg::QueryBegin(span, querymsg)) => {
                        let start = Instant::now();
                        frame.parse_st = ParseState::HaveQuery
                            (Query { span, msg: querymsg }, start)
                    },
                    (ParseState::Clear,
                     ProfileQueriesMsg::CacheHit) => {
                        panic!("parse error: unexpected CacheHit; expected QueryBegin")
                    },
                    (ParseState::Clear,
                     ProfileQueriesMsg::ProviderBegin) => {
                        panic!("parse error: expected QueryBegin before beginning a provider")
                    },
                    (ParseState::Clear,
                     ProfileQueriesMsg::ProviderEnd) => {
                        let provider_extent = frame.traces;
                        match stack.pop() {
                            None =>
                                panic!("parse error: expected a stack frame; found an empty stack"),
                            Some(old_frame) => {
                                match old_frame.parse_st {
                                    ParseState::HaveQuery(q, start) => {
                                        let duration = start.elapsed();
                                        frame = StackFrame{
                                            parse_st: ParseState::Clear,
                                            traces: old_frame.traces
                                        };
                                        let dur_extent = total_duration(&provider_extent);
                                        let trace = Rec {
                                            effect: Effect::QueryBegin(q, CacheCase::Miss),
                                            extent: Box::new(provider_extent),
                                            start: start,
                                            dur_self: duration - dur_extent,
                                            dur_total: duration,
                                        };
                                        frame.traces.push( trace );
                                    },
                                    _ => panic!("internal parse error: malformed parse stack")
                                }
                            }
                        }
                    },
                    (ParseState::Clear,
                     ProfileQueriesMsg::TimeBegin(msg)) => {
                        let start = Instant::now();
                        frame.parse_st = ParseState::HaveTimeBegin(msg, start);
                        stack.push(frame);
                        frame = StackFrame{parse_st: ParseState::Clear, traces: vec![]};
                    },
                    (_, ProfileQueriesMsg::TimeBegin(_)) => {
                        panic!("parse error; did not expect time begin here");
                    },
                    (ParseState::Clear,
                     ProfileQueriesMsg::TimeEnd) => {
                        let provider_extent = frame.traces;
                        match stack.pop() {
                            None =>
                                panic!("parse error: expected a stack frame; found an empty stack"),
                            Some(old_frame) => {
                                match old_frame.parse_st {
                                    ParseState::HaveTimeBegin(msg, start) => {
                                        let duration = start.elapsed();
                                        frame = StackFrame{
                                            parse_st: ParseState::Clear,
                                            traces: old_frame.traces
                                        };
                                        let dur_extent = total_duration(&provider_extent);
                                        let trace = Rec {
                                            effect: Effect::TimeBegin(msg),
                                            extent: Box::new(provider_extent),
                                            start: start,
                                            dur_total: duration,
                                            dur_self: duration - dur_extent,
                                        };
                                        frame.traces.push( trace );
                                    },
                                    _ => panic!("internal parse error: malformed parse stack")
                                }
                            }
                        }
                    },
                    (_, ProfileQueriesMsg::TimeEnd) => {
                        panic!("parse error")
                    },
                    (ParseState::Clear,
                     ProfileQueriesMsg::TaskBegin(key)) => {
                        let start = Instant::now();
                        frame.parse_st = ParseState::HaveTaskBegin(key, start);
                        stack.push(frame);
                        frame = StackFrame{ parse_st: ParseState::Clear, traces: vec![] };
                    },
                    (_, ProfileQueriesMsg::TaskBegin(_)) => {
                        panic!("parse error; did not expect time begin here");
                    },
                    (ParseState::Clear,
                     ProfileQueriesMsg::TaskEnd) => {
                        let provider_extent = frame.traces;
                        match stack.pop() {
                            None =>
                                panic!("parse error: expected a stack frame; found an empty stack"),
                            Some(old_frame) => {
                                match old_frame.parse_st {
                                    ParseState::HaveTaskBegin(key, start) => {
                                        let duration = start.elapsed();
                                        frame = StackFrame{
                                            parse_st: ParseState::Clear,
                                            traces: old_frame.traces
                                        };
                                        let dur_extent = total_duration(&provider_extent);
                                        let trace = Rec {
                                            effect: Effect::TaskBegin(key),
                                            extent: Box::new(provider_extent),
                                            start: start,
                                            dur_total: duration,
                                            dur_self: duration - dur_extent,
                                        };
                                        frame.traces.push( trace );
                                    },
                                    _ => panic!("internal parse error: malformed parse stack")
                                }
                            }
                        }
                    },
                    (_, ProfileQueriesMsg::TaskEnd) => {
                        panic!("parse error")
                    },
                    // Parse State: HaveQuery
                    (ParseState::HaveQuery(q,start),
                     ProfileQueriesMsg::CacheHit) => {
                        let duration = start.elapsed();
                        let trace : Rec = Rec{
                            effect: Effect::QueryBegin(q, CacheCase::Hit),
                            extent: Box::new(vec![]),
                            start: start,
                            dur_self: duration,
                            dur_total: duration,
                        };
                        frame.traces.push( trace );
                        frame.parse_st = ParseState::Clear;
                    },
                    (ParseState::HaveQuery(_, _),
                     ProfileQueriesMsg::ProviderBegin) => {
                        stack.push(frame);
                        frame = StackFrame{ parse_st: ParseState::Clear, traces: vec![] };
                    },

                    // Parse errors:

                    (ParseState::HaveQuery(q, _),
                     ProfileQueriesMsg::ProviderEnd) => {
                        panic!("parse error: unexpected ProviderEnd; \
                                expected something else to follow BeginQuery for {:?}", q)
                    },
                    (ParseState::HaveQuery(q1, _),
                     ProfileQueriesMsg::QueryBegin(span2, querymsg2)) => {
                        panic!("parse error: unexpected QueryBegin; \
                                earlier query is unfinished: {:?} and now {:?}",
                               q1, Query{span:span2, msg: querymsg2})
                    },
                    (ParseState::HaveTimeBegin(_, _), _) => {
                        unreachable!()
                    },
                    (ParseState::HaveTaskBegin(_, _), _) => {
                        unreachable!()
                    },
                }
            }
        }
    }
}
