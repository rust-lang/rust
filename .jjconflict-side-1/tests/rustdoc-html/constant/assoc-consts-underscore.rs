pub struct Struct {
    _private: (),
}

pub trait Trait {
    //@ has assoc_consts_underscore/trait.Trait.html '//pre[@class="rust item-decl"]' \
    //      'const REQUIRED: Struct;'
    //@ !has - '//*[@id="associatedconstant.REQUIRED"]' 'const REQUIRED: Struct = _'
    //@ has - '//*[@id="associatedconstant.REQUIRED"]' 'const REQUIRED: Struct'
    const REQUIRED: Struct;
    //@ has - '//pre[@class="rust item-decl"]' 'const OPTIONAL: Struct = _;'
    //@ has - '//*[@id="associatedconstant.OPTIONAL"]' 'const OPTIONAL: Struct = _'
    const OPTIONAL: Struct = Struct { _private: () };
}

impl Trait for Struct {
    //@ !has assoc_consts_underscore/struct.Struct.html '//*[@id="associatedconstant.REQUIRED"]' \
    //      'const REQUIRED: Struct = _'
    //@ has - '//*[@id="associatedconstant.REQUIRED"]' 'const REQUIRED: Struct'
    const REQUIRED: Struct = Struct { _private: () };
    //@ !has - '//*[@id="associatedconstant.OPTIONAL"]' 'const OPTIONAL: Struct = _'
    //@ has - '//*[@id="associatedconstant.OPTIONAL"]' 'const OPTIONAL: Struct'
    const OPTIONAL: Struct = Struct { _private: () };
}

impl Struct {
    //@ !has - '//*[@id="associatedconstant.INHERENT"]' 'const INHERENT: Struct = _'
    //@ has - '//*[@id="associatedconstant.INHERENT"]' 'const INHERENT: Struct'
    pub const INHERENT: Struct = Struct { _private: () };
}
