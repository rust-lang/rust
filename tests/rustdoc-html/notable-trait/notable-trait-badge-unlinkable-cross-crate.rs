//@ aux-build:notable-dep.rs

#![crate_name = "foo"]

extern crate notable_dep;

// A notable trait from a dependency that was compiled but not documented is
// unlinkable: the badge is still emitted but rendered as plain text.
use notable_dep::Spaceship;

//@ has 'foo/struct.Rocket.html'
//@ has - '//div[@class="notable-trait-badge-container"]/a' 'Spaceship'
// We ensure that the `tabindex` is set so you can focus the `span`.
//@ has - '//div[@class="notable-trait-badge-container"]/span/@tabindex' '0'
pub struct Rocket;
impl Spaceship for Rocket {}
