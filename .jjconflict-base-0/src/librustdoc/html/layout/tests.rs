#[test]
fn test_may_remove_crossorigin() {
    use super::may_remove_crossorigin;

    assert!(may_remove_crossorigin("font.woff2"));
    assert!(may_remove_crossorigin("/font.woff2"));
    assert!(may_remove_crossorigin("./font.woff2"));
    assert!(may_remove_crossorigin(":D/font.woff2"));
    assert!(may_remove_crossorigin("../font.woff2"));

    assert!(!may_remove_crossorigin("//example.com/static.files"));
    assert!(!may_remove_crossorigin("http://example.com/static.files"));
    assert!(!may_remove_crossorigin("https://example.com/static.files"));
    assert!(!may_remove_crossorigin("https://example.com:8080/static.files"));

    assert!(!may_remove_crossorigin("ftp://example.com/static.files"));
    assert!(!may_remove_crossorigin("blob:http://example.com/static.files"));
    assert!(!may_remove_crossorigin("javascript:alert('Hello, world!')"));
    assert!(!may_remove_crossorigin("//./C:"));
    assert!(!may_remove_crossorigin("file:////C:"));
    assert!(!may_remove_crossorigin("file:///./C:"));
    assert!(!may_remove_crossorigin("data:,Hello%2C%20World%21"));
    assert!(!may_remove_crossorigin("hi...:hello"));
}
