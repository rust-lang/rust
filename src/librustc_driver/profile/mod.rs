// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::util::common::{ProfQDumpParams, ProfileQueriesMsg, profq_msg, profq_set_chan};
use std::sync::mpsc::{Receiver};
use std::io::{Write};

pub mod trace;

/// begin a profile thread, if not already running
pub fn begin() {
    use std::thread;
    use std::sync::mpsc::{channel};
    let (tx, rx) = channel();
    if profq_set_chan(tx) {
        thread::spawn(move||profile_queries_thread(rx));
    }
}

/// dump files with profiling information to the given base path, and
/// wait for this dump to complete.
///
/// wraps the RPC (send/recv channel logic) of requesting a dump.
pub fn dump(path:String) {
    use std::sync::mpsc::{channel};
    let (tx, rx) = channel();
    let params = ProfQDumpParams{
        path, ack:tx,
        // FIXME: Add another compiler flag to toggle whether this log
        // is written; false for now
        dump_profq_msg_log:false,
    };
    profq_msg(ProfileQueriesMsg::Dump(params));
    let _ = rx.recv().unwrap();
}

// profiling thread; retains state (in local variables) and dump traces, upon request.
fn profile_queries_thread(r:Receiver<ProfileQueriesMsg>) {
    use self::trace::*;
    use std::fs::File;
    use std::time::{Instant};

    let mut profq_msgs : Vec<ProfileQueriesMsg> = vec![];
    let mut frame : StackFrame = StackFrame{ parse_st:ParseState::NoQuery, traces:vec![] };
    let mut stack : Vec<StackFrame> = vec![];
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
                assert!(stack.len() == 0);
                assert!(frame.parse_st == trace::ParseState::NoQuery);
                {
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

                    write!(html_file, "<html>\n").unwrap();
                    write!(html_file,
                           "<head>\n<link rel=\"stylesheet\" type=\"text/css\" href=\"{}\">\n",
                           "profile_queries.css").unwrap();
                    write!(html_file, "<style>\n").unwrap();
                    trace::write_style(&mut html_file);
                    write!(html_file, "</style>\n").unwrap();
                    write!(html_file, "</head>\n").unwrap();
                    write!(html_file, "<body>\n").unwrap();
                    trace::write_traces(&mut html_file, &mut counts_file, &frame.traces);
                    write!(html_file, "</body>\n</html>\n").unwrap();

                    // Tell main thread that we are done, e.g., so it can exit
                    params.ack.send(()).unwrap();
                }
                continue
            }
            // Actual query message:
            msg => {
                // Record msg in our log
                profq_msgs.push(msg.clone());
                // Respond to the message, knowing that we've already handled Halt and Dump, above.
                match (frame.parse_st.clone(), msg) {
                    (_,ProfileQueriesMsg::Halt) => unreachable!(),
                    (_,ProfileQueriesMsg::Dump(_)) => unreachable!(),

                    // Parse State: NoQuery
                    (ParseState::NoQuery,
                     ProfileQueriesMsg::QueryBegin(span,querymsg)) => {
                        let start = Instant::now();
                        frame.parse_st = ParseState::HaveQuery
                            (Query{span:span, msg:querymsg}, start)
                    },
                    (ParseState::NoQuery,
                     ProfileQueriesMsg::CacheHit) => {
                        panic!("parse error: unexpected CacheHit; expected QueryBegin")
                    },
                    (ParseState::NoQuery,
                     ProfileQueriesMsg::ProviderBegin) => {
                        panic!("parse error: expected QueryBegin before beginning a provider")
                    },
                    (ParseState::NoQuery,
                     ProfileQueriesMsg::ProviderEnd) => {
                        let provider_extent = frame.traces;
                        match stack.pop() {
                            None =>
                                panic!("parse error: expected a stack frame; found an empty stack"),
                            Some(old_frame) => {
                                match old_frame.parse_st {
                                    ParseState::NoQuery =>
                                        panic!("parse error: expected a stack frame for a query"),
                                    ParseState::HaveQuery(q,start) => {
                                        let duration = start.elapsed();
                                        frame = StackFrame{
                                            parse_st:ParseState::NoQuery,
                                            traces:old_frame.traces
                                        };
                                        let trace = Rec {
                                            effect: Effect::QueryBegin(q, CacheCase::Miss),
                                            extent: Box::new(provider_extent),
                                            start: start,
                                            duration: duration,
                                        };
                                        frame.traces.push( trace );
                                    }
                                }
                            }
                        }
                    }

                    // Parse State: HaveQuery
                    (ParseState::HaveQuery(q,start),
                     ProfileQueriesMsg::CacheHit) => {
                        let duration = start.elapsed();
                        let trace : Rec = Rec{
                            effect: Effect::QueryBegin(q, CacheCase::Hit),
                            extent: Box::new(vec![]),
                            start: start,
                            duration: duration,
                        };
                        frame.traces.push( trace );
                        frame.parse_st = ParseState::NoQuery;
                    },
                    (ParseState::HaveQuery(_,_),
                     ProfileQueriesMsg::ProviderBegin) => {
                        stack.push(frame);
                        frame = StackFrame{parse_st:ParseState::NoQuery, traces:vec![]};
                    },
                    (ParseState::HaveQuery(q,_),
                     ProfileQueriesMsg::ProviderEnd) => {
                        panic!("parse error: unexpected ProviderEnd; \
                                expected something else to follow BeginQuery for {:?}", q)
                    },
                    (ParseState::HaveQuery(q1,_),
                     ProfileQueriesMsg::QueryBegin(span2,querymsg2)) => {
                        panic!("parse error: unexpected QueryBegin; \
                                earlier query is unfinished: {:?} and now {:?}",
                               q1, Query{span:span2, msg:querymsg2})
                    },
                }
            }
        }
    }
}
