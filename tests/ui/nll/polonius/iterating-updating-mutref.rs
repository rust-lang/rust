// These are some examples of iterating through and updating a mutable ref, similar in spirit to the
// linked-list-like pattern of #46859/#48001 where the polonius alpha analysis shows imprecision,
// unlike the datalog implementation.
//
// They differ in that after the loans prior to the loop are either not live after the loop, or with
// control flow and outlives relationships that are simple enough for the reachability
// approximation. They're thus accepted by the alpha analysis, like NLLs did for the simplest cases
// of flow-sensitivity.

//@ ignore-compare-mode-polonius (explicit revisions)
//@ revisions: nll polonius legacy
//@ [nll] known-bug: #46859
//@ [polonius] check-pass
//@ [polonius] compile-flags: -Z polonius=next
//@ [legacy] check-pass
//@ [legacy] compile-flags: -Z polonius=legacy

// The #46859 OP
struct List<T> {
    value: T,
    next: Option<Box<List<T>>>,
}

fn to_refs<T>(mut list: &mut List<T>) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut list.value);
        if let Some(n) = list.next.as_mut() {
            list = n;
        } else {
            return result;
        }
    }
}

// A similar construction, where paths in the constraint graph are also clearly terminating, so it's
// fine even for NLLs.
fn to_refs2<T>(mut list: &mut List<T>) -> Vec<&mut T> {
    let mut result = vec![];
    loop {
        result.push(&mut list.value);
        if let Some(n) = list.next.as_mut() {
            list = n;
        } else {
            break;
        }
    }

    result
}

// Another MCVE from the same issue, but was rejected by NLLs.
pub struct Decoder {
    buf_read: BufRead,
}

impl Decoder {
    // NLLs fail here
    pub fn next<'a>(&'a mut self) -> &'a str {
        loop {
            let buf = self.buf_read.fill_buf();
            if let Some(s) = decode(buf) {
                return s;
            }
            // loop to get more input data

            // At this point `buf` is not used anymore.
            // With NLL I would expect the borrow to end here,
            // such that `self.buf_read` is not borrowed anymore
            // by the time we start the next loop iteration.
        }
    }
}

struct BufRead;

impl BufRead {
    fn fill_buf(&mut self) -> &[u8] {
        unimplemented!()
    }
}

fn decode(_: &[u8]) -> Option<&str> {
    unimplemented!()
}

fn main() {}
