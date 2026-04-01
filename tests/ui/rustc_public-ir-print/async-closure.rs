//@ compile-flags: -Z unpretty=stable-mir --crate-type lib -C panic=abort -Zmir-opt-level=0
//@ check-pass
//@ only-64bit
//@ edition: 2024
//@ needs-unwind unwind edges are different with panic=abort

pub fn foo() {
    let y = 0;
    let x = async || {
        let y = y;
    };
}
