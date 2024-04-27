// Parse `cfg_attr` with varying numbers of attributes and trailing commas

// Completely empty `cfg_attr` input
#[cfg_attr()] //~ error: malformed `cfg_attr` attribute input
struct NoConfigurationPredicate;

// Zero attributes, zero trailing comma (comma manatory here)
#[cfg_attr(all())] //~ error: expected `,`, found end of `cfg_attr`
struct A0C0;

// Zero attributes, one trailing comma
#[cfg_attr(all(),)] // Ok
struct A0C1;

// Zero attributes, two trailing commas
#[cfg_attr(all(),,)] //~ ERROR expected identifier
struct A0C2;

// One attribute, no trailing comma
#[cfg_attr(all(), must_use)] // Ok
struct A1C0;

// One attribute, one trailing comma
#[cfg_attr(all(), must_use,)] // Ok
struct A1C1;

// One attribute, two trailing commas
#[cfg_attr(all(), must_use,,)] //~ ERROR expected identifier
struct A1C2;

// Two attributes, no trailing comma
#[cfg_attr(all(), must_use, deprecated)] // Ok
struct A2C0;

// Two attributes, one trailing comma
#[cfg_attr(all(), must_use, deprecated,)] // Ok
struct A2C1;

// Two attributes, two trailing commas
#[cfg_attr(all(), must_use, deprecated,,)] //~ ERROR expected identifier
struct A2C2;

// Wrong delimiter `[`
#[cfg_attr[all(),,]]
//~^ ERROR wrong `cfg_attr` delimiters
//~| ERROR expected identifier, found `,`
struct BracketZero;

// Wrong delimiter `{`
#[cfg_attr{all(),,}]
//~^ ERROR wrong `cfg_attr` delimiters
//~| ERROR expected identifier, found `,`
struct BraceZero;

fn main() {}
