//@ build-pass
//
// (this is deliberately *not* check-pass; I have confirmed that the bug in
// question does not replicate when one uses `cargo check` alone.)

pub enum Void {}

enum UninhabitedUnivariant {
    _Variant(Void),
}

enum UninhabitedMultivariant2 {
    _Variant(Void),
    _Warriont(Void),
}

enum UninhabitedMultivariant3 {
    _Variant(Void),
    _Warriont(Void),
    _Worrynot(Void),
}

#[repr(C)]
enum UninhabitedUnivariantC {
    _Variant(Void),
}

#[repr(i32)]
enum UninhabitedUnivariant32 {
    _Variant(Void),
}

fn main() {
    let _seed: UninhabitedUnivariant = None.unwrap();
    match _seed {
        UninhabitedUnivariant::_Variant(_x) => {}
    }

    let _seed: UninhabitedMultivariant2 = None.unwrap();
    match _seed {
        UninhabitedMultivariant2::_Variant(_x) => {}
        UninhabitedMultivariant2::_Warriont(_x) => {}
    }

    let _seed: UninhabitedMultivariant2 = None.unwrap();
    match _seed {
        UninhabitedMultivariant2::_Variant(_x) => {}
        _ => {}
    }

    let _seed: UninhabitedMultivariant2 = None.unwrap();
    match _seed {
        UninhabitedMultivariant2::_Warriont(_x) => {}
        _ => {}
    }

    let _seed: UninhabitedMultivariant3 = None.unwrap();
    match _seed {
        UninhabitedMultivariant3::_Variant(_x) => {}
        UninhabitedMultivariant3::_Warriont(_x) => {}
        UninhabitedMultivariant3::_Worrynot(_x) => {}
    }

    let _seed: UninhabitedMultivariant3 = None.unwrap();
    match _seed {
        UninhabitedMultivariant3::_Variant(_x) => {}
        _ => {}
    }

    let _seed: UninhabitedMultivariant3 = None.unwrap();
    match _seed {
        UninhabitedMultivariant3::_Warriont(_x) => {}
        _ => {}
    }

    let _seed: UninhabitedMultivariant3 = None.unwrap();
    match _seed {
        UninhabitedMultivariant3::_Worrynot(_x) => {}
        _ => {}
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
