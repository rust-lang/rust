// FIXME(workingjubilee): add revisions and generalize to other platform-specific varargs ABIs,
// preferably after the only-arch directive is enhanced with an "or pattern" syntax
// NOTE: This deliberately tests an ABI that supports varargs, so no `extern "rust-invalid"`
//@ only-x86_64

// We have to use this flag to force ABI computation of an invalid ABI
//@ compile-flags: -Clink-dead-code

// sometimes fn ptrs with varargs make layout and ABI computation ICE
// as found in https://github.com/rust-lang/rust/issues/142107

fn aapcs(f: extern "aapcs" fn(usize, ...)) {
//~^ ERROR [E0570]
// Note we DO NOT have to actually make a call to trigger the ICE!
}

fn main() {}
