#![doc(html_logo_url(light = "https://example.com/logo-light.png"))]
// Only a light logo is given, so no dark logo or `has-dark-logo` class.

//@ has logo_dark_light_only/struct.SomeStruct.html '//img[@class="logo-light"][@src="https://example.com/logo-light.png"]' ''
//@ has logo_dark_light_only/struct.SomeStruct.html '//a[@class="logo-container"]' ''
//@ !has logo_dark_light_only/struct.SomeStruct.html '//img[@class="logo-dark"]' ''
//@ !has logo_dark_light_only/struct.SomeStruct.html '//a[@class="logo-container has-dark-logo"]' ''
pub struct SomeStruct;
