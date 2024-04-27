// Format `lazy_static!`.

lazy_static! {
    static ref CONFIG_NAME_REGEX: regex::Regex =
        regex::Regex::new(r"^## `([^`]+)`").expect("Failed creating configuration pattern");
    static ref CONFIG_VALUE_REGEX: regex::Regex = regex::Regex::new(r#"^#### `"?([^`"]+)"?`"#)
        .expect("Failed creating configuration value pattern");
}

// We need to be able to format `lazy_static!` without known syntax.
lazy_static!(xxx, yyyy, zzzzz);

lazy_static! {}

// #2354
lazy_static! {
    pub static ref Sbase64_encode_string: ::lisp::LispSubrRef = {
        let subr = ::remacs_sys::Lisp_Subr {
            header: ::remacs_sys::Lisp_Vectorlike_Header {
                size: ((::remacs_sys::PseudovecType::PVEC_SUBR as ::libc::ptrdiff_t)
                    << ::remacs_sys::PSEUDOVECTOR_AREA_BITS),
            },
            function: self::Fbase64_encode_string as *const ::libc::c_void,
            min_args: 1i16,
            max_args: 2i16,
            symbol_name: (b"base64-encode-string\x00").as_ptr() as *const ::libc::c_char,
            intspec: ::std::ptr::null(),
            doc: ::std::ptr::null(),
            lang: ::remacs_sys::Lisp_Subr_Lang_Rust,
        };
        unsafe {
            let ptr = ::remacs_sys::xmalloc(::std::mem::size_of::<::remacs_sys::Lisp_Subr>())
                as *mut ::remacs_sys::Lisp_Subr;
            ::std::ptr::copy_nonoverlapping(&subr, ptr, 1);
            ::std::mem::forget(subr);
            ::lisp::ExternalPtr::new(ptr)
        }
    };
}

lazy_static! {
    static ref FOO: HashMap<
        String,
        (
            &'static str,
            fn(Foo) -> Result<Box<Bar>, Either<FooError, BarError>>
        ),
    > = HashMap::new();
}
