//@ compile-flags: -Z unpretty=stable-mir --crate-type lib -C panic=abort
//@ check-pass
//@ only-x86_64
//@ edition: 2024
//@ needs-unwind unwind edges are different with panic=abort

pub fn foo() {
    let y = 0;
    let x = async || {
        let y = y;
    };
}
