#![warn(clippy::const_is_empty)]

fn test_literal() {
    if "".is_empty() {
        //~^ERROR: this expression always evaluates to true
    }
    if "foobar".is_empty() {
        //~^ERROR: this expression always evaluates to false
    }
}

fn test_byte_literal() {
    if b"".is_empty() {
        //~^ERROR: this expression always evaluates to true
    }
    if b"foobar".is_empty() {
        //~^ERROR: this expression always evaluates to false
    }
}

fn test_no_mut() {
    let mut empty = "";
    if empty.is_empty() {
        // No lint because it is mutable
    }
}

fn test_propagated() {
    let empty = "";
    let non_empty = "foobar";
    let empty2 = empty;
    let non_empty2 = non_empty;
    if empty2.is_empty() {
        //~^ERROR: this expression always evaluates to true
    }
    if non_empty2.is_empty() {
        //~^ERROR: this expression always evaluates to false
    }
}

const EMPTY_STR: &str = "";
const NON_EMPTY_STR: &str = "foo";
const EMPTY_BSTR: &[u8] = b"";
const NON_EMPTY_BSTR: &[u8] = b"foo";
const EMPTY_U8_SLICE: &[u8] = &[];
const NON_EMPTY_U8_SLICE: &[u8] = &[1, 2];
const EMPTY_SLICE: &[u32] = &[];
const NON_EMPTY_SLICE: &[u32] = &[1, 2];
const NON_EMPTY_SLICE_REPEAT: &[u32] = &[1; 2];
const EMPTY_ARRAY: [u32; 0] = [];
const EMPTY_ARRAY_REPEAT: [u32; 0] = [1; 0];
const NON_EMPTY_ARRAY: [u32; 2] = [1, 2];
const NON_EMPTY_ARRAY_REPEAT: [u32; 2] = [1; 2];
const EMPTY_REF_ARRAY: &[u32; 0] = &[];
const NON_EMPTY_REF_ARRAY: &[u32; 3] = &[1, 2, 3];

fn test_from_const() {
    let _ = EMPTY_STR.is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = NON_EMPTY_STR.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = EMPTY_BSTR.is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = NON_EMPTY_BSTR.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = EMPTY_ARRAY.is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = EMPTY_ARRAY_REPEAT.is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = EMPTY_U8_SLICE.is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = NON_EMPTY_U8_SLICE.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = NON_EMPTY_ARRAY.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = NON_EMPTY_ARRAY_REPEAT.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = EMPTY_REF_ARRAY.is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = NON_EMPTY_REF_ARRAY.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = EMPTY_SLICE.is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = NON_EMPTY_SLICE.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = NON_EMPTY_SLICE_REPEAT.is_empty();
    //~^ ERROR: this expression always evaluates to false
}

fn main() {
    let value = "foobar";
    let _ = value.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let x = value;
    let _ = x.is_empty();
    //~^ ERROR: this expression always evaluates to false
    let _ = "".is_empty();
    //~^ ERROR: this expression always evaluates to true
    let _ = b"".is_empty();
    //~^ ERROR: this expression always evaluates to true
}
