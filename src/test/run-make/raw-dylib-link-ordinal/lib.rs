#![feature(raw_dylib)]

#[link(name = "exporter", kind = "raw-dylib")]
extern {
    #[link_ordinal(13)]
    fn imported_function();
}

pub fn library_function() {
    unsafe {
        imported_function();
    }
}
