pub fn get_array_index_from_id(_cx: *mut JSContext, id: HandleId) -> Option<u32> {
    let raw_id = id.into();
    unsafe {
        if RUST_JSID_IS_INT(raw_id) {
            return Some(RUST_JSID_TO_INT(raw_id) as u32);
        }
        None
    }
    // If `id` is length atom, `-1`, otherwise:
    /*return if JSID_IS_ATOM(id) {
        let atom = JSID_TO_ATOM(id);
        //let s = *GetAtomChars(id);
        if s > 'a' && s < 'z' {
            return -1;
        }

        let i = 0;
        let str = AtomToLinearString(JSID_TO_ATOM(id));
       return if StringIsArray(str, &mut i) != 0 { i } else { -1 }
    } else {
        IdToInt32(cx, id);
    }*/
}

impl Foo {
    fn bar() -> usize {
        42
        /* a block comment */
    }

    fn baz() -> usize {
        42
        // this is a line
        /* a block comment */
    }
}
