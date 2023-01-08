// run-rustfix
#![deny(rustdoc::unprefixed_html_class)]

/// Test with <div class="evil"></div>
//~^ ERROR unprefixed HTML `class` attribute
pub struct BasicBad;

/// Test with <div class="unprefixed_html_class_evil"></div>
pub struct BasicGood;

/// Test with <div class="unprefixed_html_class_evil evil"></div>
//~^ ERROR unprefixed HTML `class` attribute
pub struct MixedBadAndGood1;

/// Test with <div class="evil unprefixed_html_class_evil"></div>
//~^ ERROR unprefixed HTML `class` attribute
pub struct MixedBadAndGood2;

/// Test with <div class="evil genius"></div>
//~^ ERROR unprefixed HTML `class` attribute
//~^^ ERROR unprefixed HTML `class` attribute
pub struct DoubleBad;

// `stab` is currently the only class name that's allowed unprefixed.

/// Test with <div class="stab"></div>
/// Test with <div class="stab deprecated"></div>
/// Test with <div class="stab portability"></div>
/// Test with <div class="deprecated stab"></div>
/// Test with <div class="portability stab"></div>
/// Test with <div class="deprecated stab unprefixed_html_class_extras"></div>
/// Test with <div class="portability stab"></div>
pub struct Stab;

/// Test with <div class="deprecated"></div>
//~^ ERROR unprefixed HTML `class` attribute
pub struct StandaloneDeprecatedIsBad;

/// Test with <div class="portability"></div>
//~^ ERROR unprefixed HTML `class` attribute
pub struct StandalonePortabilityIsBad;

/// Test unquoted: <div class=stab></div>
pub struct UnquotedStab;

/// Test unquoted: <div class=bad></div>
//~^ ERROR unprefixed HTML `class` attribute
pub struct UnquotedBad;

/// Test unquoted: <div class=bad title="Whatever"></div>
//~^ ERROR unprefixed HTML `class` attribute
pub struct UnquotedBadMixed1;

/// Test unquoted: <div disabled class=bad></div>
//~^ ERROR unprefixed HTML `class` attribute
pub struct UnquotedBadMixed2;
