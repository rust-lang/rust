#![allow(clippy::redundant_closure)]

fn main() {
    let _ = (0..4).filter_map(|x| if x > 1 { Some(x) } else { None });
    //~^ unnecessary_filter_map

    let _ = (0..4).filter_map(|x| {
        //~^ unnecessary_filter_map

        if x > 1 {
            return Some(x);
        };
        None
    });
    let _ = (0..4).filter_map(|x| match x {
        //~^ unnecessary_filter_map
        0 | 1 => None,
        _ => Some(x),
    });

    let _ = (0..4).filter_map(|x| Some(x + 1));
    //~^ unnecessary_filter_map

    let _ = (0..4).filter_map(i32::checked_abs);

    let _ = (0..4).filter_map(Some);

    let _ = vec![Some(10), None].into_iter().filter_map(|x| Some(x));
    //~^ unnecessary_filter_map
}

fn filter_map_none_changes_item_type() -> impl Iterator<Item = bool> {
    "".chars().filter_map(|_| None)
}

// https://github.com/rust-lang/rust-clippy/issues/4433#issue-483920107
mod comment_483920107 {
    enum Severity {
        Warning,
        Other,
    }

    struct ServerError;

    impl ServerError {
        fn severity(&self) -> Severity {
            Severity::Warning
        }
    }

    struct S {
        warnings: Vec<ServerError>,
    }

    impl S {
        fn foo(&mut self, server_errors: Vec<ServerError>) {
            #[allow(unused_variables)]
            let errors: Vec<ServerError> = server_errors
                .into_iter()
                .filter_map(|se| match se.severity() {
                    Severity::Warning => {
                        self.warnings.push(se);
                        None
                    },
                    _ => Some(se),
                })
                .collect();
        }
    }
}

// https://github.com/rust-lang/rust-clippy/issues/4433#issuecomment-611006622
mod comment_611006622 {
    struct PendingRequest {
        reply_to: u8,
        token: u8,
        expires: u8,
        group_id: u8,
    }

    enum Value {
        Null,
    }

    struct Node;

    impl Node {
        fn send_response(&self, _reply_to: u8, _token: u8, _value: Value) -> &Self {
            self
        }
        fn on_error_warn(&self) -> &Self {
            self
        }
    }

    struct S {
        pending_requests: Vec<PendingRequest>,
    }

    impl S {
        fn foo(&mut self, node: Node, now: u8, group_id: u8) {
            // "drain_filter"
            self.pending_requests = self
                .pending_requests
                .drain(..)
                .filter_map(|pending| {
                    if pending.expires <= now {
                        return None; // Expired, remove
                    }

                    if pending.group_id == group_id {
                        // Matched - reuse strings and remove
                        node.send_response(pending.reply_to, pending.token, Value::Null)
                            .on_error_warn();
                        None
                    } else {
                        // Keep waiting
                        Some(pending)
                    }
                })
                .collect();
        }
    }
}

// https://github.com/rust-lang/rust-clippy/issues/4433#issuecomment-621925270
// This extrapolation doesn't reproduce the false positive. Additional context seems necessary.
mod comment_621925270 {
    struct Signature(u8);

    fn foo(sig_packets: impl Iterator<Item = Result<Signature, ()>>) -> impl Iterator<Item = u8> {
        sig_packets.filter_map(|res| match res {
            Ok(Signature(sig_packet)) => Some(sig_packet),
            _ => None,
        })
    }
}

// https://github.com/rust-lang/rust-clippy/issues/4433#issuecomment-1052978898
mod comment_1052978898 {
    #![allow(clippy::redundant_closure)]

    pub struct S(u8);

    impl S {
        pub fn consume(self) {
            println!("yum");
        }
    }

    pub fn filter_owned() -> impl Iterator<Item = S> {
        (0..10).map(|i| S(i)).filter_map(|s| {
            if s.0 & 1 == 0 {
                s.consume();
                None
            } else {
                Some(s)
            }
        })
    }
}

fn issue11260() {
    // #11260 is about unnecessary_find_map, but the fix also kind of applies to
    // unnecessary_filter_map
    let _x = std::iter::once(1).filter_map(|n| (n > 1).then_some(n));
    //~^ unnecessary_filter_map
}
