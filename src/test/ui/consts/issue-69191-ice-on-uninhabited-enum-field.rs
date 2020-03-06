// build-pass
//
// (this is deliberately *not* check-pass; I have confirmed that the bug in
// question does not replicate when one uses `cargo check` alone.)

pub enum Void {}

enum UninhabitedUnivariant { _Variant(Void), }

#[repr(C)]
enum UninhabitedUnivariantC { _Variant(Void), }

#[repr(i32)]
enum UninhabitedUnivariant32 { _Variant(Void), }

fn main() {
    let _seed: UninhabitedUnivariant = None.unwrap();
    match _seed {
        UninhabitedUnivariant::_Variant(_x) => {}
    }

    let _seed: UninhabitedUnivariantC = None.unwrap();
    match _seed {
        UninhabitedUnivariantC::_Variant(_x) => {}
    }

    let _seed: UninhabitedUnivariant32 = None.unwrap();
    match _seed {
        UninhabitedUnivariant32::_Variant(_x) => {}
    }
}
