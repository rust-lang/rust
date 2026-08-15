//@ compile-flags: --document-private-items

//@ !has const_underscore/constant._.html
const _: () = {
    #[no_mangle]
    extern "C" fn implementation_detail() {}
};
