//@ aux-build:notable-dep.rs

#![crate_name = "foo"]

extern crate notable_dep;

// A notable trait from a dependency that was compiled but not documented is
// unlinkable: the badge is still emitted but rendered as plain text.
use notable_dep::Spaceship;

//@ has 'foo/struct.Rocket.html'
// The badge is present...
//@ has - '//div[@class="notable-trait-badge-container"]/a' 'Spaceship'
// ...but unlinked: no badge carries an `href`.
//@ count - '//div[@class="notable-trait-badge-container"]/a[@href]' 0
pub struct Rocket;
impl Spaceship for Rocket {}
