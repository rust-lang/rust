macro_rules! fuzz_target {
    (|$data:ident: &[u8]| $body:block) => {};
}

fuzz_target!(|data: &[u8]| {
    if let Ok(app_img) = AppImage::parse(data) {
        if let Ok(app_img) =
            app_img.sign_for_secureboot(include_str!("../../test-data/signing-key"))
        {
            assert!(app_img.is_signed());
            Gbl::from_app_image(app_img).to_bytes();
        }
    }
});
