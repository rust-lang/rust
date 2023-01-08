// run-rustfix
#![deny(rustdoc::unprefixed_html_id)]

/// Test with <div id="evil"></div>
//~^ ERROR unprefixed HTML `id` attribute
pub struct BasicBad;

/// Test with <div id="unprefixed_html_id_evil"></div>
pub struct BasicGood;

// `stab` is not allowed as a special ID.

/// Test with <div id="stab"></div>
//~^ ERROR unprefixed HTML `id` attribute
pub struct StabIsNotAnId;

/// Test unquoted: <div id=stab></div>
//~^ ERROR unprefixed HTML `id` attribute
pub struct UnquotedStab;

/// Test unquoted: <div id=bad title="Whatever"></div>
//~^ ERROR unprefixed HTML `id` attribute
pub struct UnquotedBadMixed1;

/// Test unquoted: <div disabled id=bad></div>
//~^ ERROR unprefixed HTML `id` attribute
pub struct UnquotedBadMixed2;
