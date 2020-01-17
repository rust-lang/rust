// check-pass
// Verifies that our fallback-UB check doesn't trigger
// on correct code
#![feature(never_type)]
#![feature(never_type_fallback)]

fn foo(e: !) -> Box<dyn std::error::Error> {
    Box::<_>::new(e)
}

fn main() {}
