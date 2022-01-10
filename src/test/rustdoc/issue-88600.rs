// This test ensure that #[doc(hidden)] is applied correctly in enum variant fields.

// Denotes a field which should be hidden.
pub struct H;

// Denotes a field which should not be hidden (shown).
pub struct S;

// @has issue_88600/enum.FooEnum.html
pub enum FooEnum {
    // @has - '//*[@id="variant.HiddenTupleItem"]//code' 'HiddenTupleItem(_)'
    // @count - '//*[@id="variant.HiddenTupleItem.field.0"]' 0
    HiddenTupleItem(#[doc(hidden)] H),
    // @has - '//*[@id="variant.MultipleHidden"]//code' 'MultipleHidden(_, _)'
    // @count - '//*[@id="variant.MultipleHidden.field.0"]' 0
    // @count - '//*[@id="variant.MultipleHidden.field.1"]' 0
    MultipleHidden(#[doc(hidden)] H, #[doc(hidden)] H),
    // @has - '//*[@id="variant.MixedHiddenFirst"]//code' 'MixedHiddenFirst(_, S)'
    // @count - '//*[@id="variant.MixedHiddenFirst.field.0"]' 0
    // @has - '//*[@id="variant.MixedHiddenFirst.field.1"]' '1: S'
    MixedHiddenFirst(#[doc(hidden)] H, /** dox */ S),
    // @has - '//*[@id="variant.MixedHiddenLast"]//code' 'MixedHiddenLast(S, _)'
    // @has - '//*[@id="variant.MixedHiddenLast.field.0"]' '0: S'
    // @count - '//*[@id="variant.MixedHiddenLast.field.1"]' 0
    MixedHiddenLast(/** dox */ S, #[doc(hidden)] H),
    // @has - '//*[@id="variant.HiddenStruct"]//code' 'HiddenStruct'
    // @count - '//*[@id="variant.HiddenStruct.field.h"]' 0
    // @has - '//*[@id="variant.HiddenStruct.field.s"]' 's: S'
    HiddenStruct {
        #[doc(hidden)]
        h: H,
        /// dox
        s: S,
    },
}
