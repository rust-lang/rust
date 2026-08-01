#![doc(html_logo_url(light = "https://example.com/logo-light.png",
    dark = "https://example.com/logo-dark.png"))]
// Both logos are provided, so the `has-dark-logo` class must be present.

//@ has logo_dark/struct.SomeStruct.html '//img[@class="logo-light"][@src="https://example.com/logo-light.png"]' ''
//@ has logo_dark/struct.SomeStruct.html '//img[@class="logo-dark"][@src="https://example.com/logo-dark.png"]' ''
//@ has logo_dark/struct.SomeStruct.html '//a[@class="logo-container has-dark-logo"]' ''
pub struct SomeStruct;
