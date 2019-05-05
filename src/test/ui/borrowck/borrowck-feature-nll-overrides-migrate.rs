// This is a test that the `#![feature(nll)]` opt-in overrides the
// migration mode. The intention here is to emulate the goal behavior
// that `--edition 2018` effects on borrowck (modeled here by `-Z
// borrowck=migrate`) are themselves overridden by the
// `#![feature(nll)]` opt-in.
//
// Therefore, for developer convenience, under `#[feature(nll)]` the
// NLL checks will be emitted as errors *even* in the presence of `-Z
// borrowck=migrate`.

// revisions: zflag edition
// [zflag]compile-flags: -Z borrowck=migrate
// [edition]edition:2018

#![feature(nll)]

fn main() {
    match Some(&4) {
        None => {},
        ref mut foo
            if {
                (|| { let bar = foo; bar.take() })();
                //[zflag]~^ ERROR cannot move out of `foo` in pattern guard [E0507]
                //[edition]~^^ ERROR cannot move out of `foo` in pattern guard [E0507]
                false
            } => {},
        Some(ref _s) => println!("Note this arm is bogus; the `Some` became `None` in the guard."),
        _ => println!("Here is some supposedly unreachable code."),
    }
}
