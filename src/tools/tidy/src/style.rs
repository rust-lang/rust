//! Tidy check to enforce various stylistic guidelines on the Rust codebase.
//!
//! Example checks are:
//!
//! * No lines over 100 characters (in non-Rust files).
//! * No files with over 3000 lines (in non-Rust files).
//! * No tabs.
//! * No trailing whitespace.
//! * No CR characters.
//! * No `TODO` or `XXX` directives.
//! * No unexplained ` ```ignore ` or ` ```rust,ignore ` doc tests.
//!
//! Note that some of these rules are excluded from Rust files because we enforce rustfmt. It is
//! preferable to be formatted rather than tidy-clean.
//!
//! A number of these checks can be opted-out of with various directives of the form:
//! `// ignore-tidy-CHECK-NAME`.
// ignore-tidy-dbg

use crate::walk::{filter_dirs, walk};
use regex::{Regex, RegexSet};
use std::{ffi::OsStr, path::Path};

/// Error code markdown is restricted to 80 columns because they can be
/// displayed on the console with --example.
const ERROR_CODE_COLS: usize = 80;
const COLS: usize = 100;
const GOML_COLS: usize = 120;

const LINES: usize = 3000;

const UNEXPLAINED_IGNORE_DOCTEST_INFO: &str = r#"unexplained "```ignore" doctest; try one:

* make the test actually pass, by adding necessary imports and declarations, or
* use "```text", if the code is not Rust code, or
* use "```compile_fail,Ennnn", if the code is expected to fail at compile time, or
* use "```should_panic", if the code is expected to fail at run time, or
* use "```no_run", if the code should type-check but not necessary linkable/runnable, or
* explain it like "```ignore (cannot-test-this-because-xxxx)", if the annotation cannot be avoided.

"#;

const LLVM_UNREACHABLE_INFO: &str = r"\
C++ code used llvm_unreachable, which triggers undefined behavior
when executed when assertions are disabled.
Use llvm::report_fatal_error for increased robustness.";

const DOUBLE_SPACE_AFTER_DOT: &str = r"\
Use a single space after dots in comments.";

const ANNOTATIONS_TO_IGNORE: &[&str] = &[
    "// @!has",
    "// @has",
    "// @matches",
    "// CHECK",
    "// EMIT_MIR",
    "// compile-flags",
    "//@ compile-flags",
    "// error-pattern",
    "//@ error-pattern",
    "// gdb",
    "// lldb",
    "// cdb",
    "// normalize-stderr-test",
    "//@ normalize-stderr-test",
];

// Intentionally written in decimal rather than hex
const PROBLEMATIC_CONSTS: &[u32] = &[
    184594917, 134263269, 184582629, 134250981, 184594741, 134263093, 184582453, 134250805,
    2880289470, 1269676734, 2829957822, 1219345086, 2873998014, 1263385278, 2823666366, 1213053630,
    2880277182, 1269664446, 2829945534, 1219332798, 2873985726, 1263372990, 2823654078, 1213041342,
    2880287934, 1269675198, 2829956286, 1219343550, 2873996478, 1263383742, 2823664830, 1213052094,
    2880275646, 1269662910, 2829943998, 1219331262, 2873984190, 1263371454, 2823652542, 1213039806,
    2880289422, 1269676686, 2829957774, 1219345038, 2873997966, 1263385230, 2823666318, 1213053582,
    2880277134, 1269664398, 2829945486, 1219332750, 2873985678, 1263372942, 2823654030, 1213041294,
    2880287886, 1269675150, 2829956238, 1219343502, 2873996430, 1263383694, 2823664782, 1213052046,
    2880275598, 1269662862, 2829943950, 1219331214, 2873984142, 1263371406, 2823652494, 1213039758,
    2880289459, 1269676723, 2829957811, 1219345075, 2873998003, 1263385267, 2823666355, 1213053619,
    2880277171, 1269664435, 2829945523, 1219332787, 2873985715, 1263372979, 2823654067, 1213041331,
    2880287923, 1269675187, 2829956275, 1219343539, 2873996467, 1263383731, 2823664819, 1213052083,
    2880275635, 1269662899, 2829943987, 1219331251, 2873984179, 1263371443, 2823652531, 1213039795,
    2880289411, 1269676675, 2829957763, 1219345027, 2873997955, 1263385219, 2823666307, 1213053571,
    2880277123, 1269664387, 2829945475, 1219332739, 2873985667, 1263372931, 2823654019, 1213041283,
    2880287875, 1269675139, 2829956227, 1219343491, 2873996419, 1263383683, 2823664771, 1213052035,
    2880275587, 1269662851, 2829943939, 1219331203, 2873984131, 1263371395, 2823652483, 1213039747,
    2881141438, 1270528702, 2830809790, 1220197054, 2877995710, 1267382974, 2827664062, 1217051326,
    2880748222, 1270135486, 2830416574, 1219803838, 2877602494, 1266989758, 2827270846, 1216658110,
    2881129150, 1270516414, 2830797502, 1220184766, 2877983422, 1267370686, 2827651774, 1217039038,
    2880735934, 1270123198, 2830404286, 1219791550, 2877590206, 1266977470, 2827258558, 1216645822,
    2881139902, 1270527166, 2830808254, 1220195518, 2877994174, 1267381438, 2827662526, 1217049790,
    2880746686, 1270133950, 2830415038, 1219802302, 2877600958, 1266988222, 2827269310, 1216656574,
    2881127614, 1270514878, 2830795966, 1220183230, 2877981886, 1267369150, 2827650238, 1217037502,
    2880734398, 1270121662, 2830402750, 1219790014, 2877588670, 1266975934, 2827257022, 1216644286,
    2881141390, 1270528654, 2830809742, 1220197006, 2877995662, 1267382926, 2827664014, 1217051278,
    2880748174, 1270135438, 2830416526, 1219803790, 2877602446, 1266989710, 2827270798, 1216658062,
    2881129102, 1270516366, 2830797454, 1220184718, 2877983374, 1267370638, 2827651726, 1217038990,
    2880735886, 1270123150, 2830404238, 1219791502, 2877590158, 1266977422, 2827258510, 1216645774,
    2881139854, 1270527118, 2830808206, 1220195470, 2877994126, 1267381390, 2827662478, 1217049742,
    2880746638, 1270133902, 2830414990, 1219802254, 2877600910, 1266988174, 2827269262, 1216656526,
    2881127566, 1270514830, 2830795918, 1220183182, 2877981838, 1267369102, 2827650190, 1217037454,
    2880734350, 1270121614, 2830402702, 1219789966, 2877588622, 1266975886, 2827256974, 1216644238,
    2881141427, 1270528691, 2830809779, 1220197043, 2877995699, 1267382963, 2827664051, 1217051315,
    2880748211, 1270135475, 2830416563, 1219803827, 2877602483, 1266989747, 2827270835, 1216658099,
    2881129139, 1270516403, 2830797491, 1220184755, 2877983411, 1267370675, 2827651763, 1217039027,
    2880735923, 1270123187, 2830404275, 1219791539, 2877590195, 1266977459, 2827258547, 1216645811,
    2881139891, 1270527155, 2830808243, 1220195507, 2877994163, 1267381427, 2827662515, 1217049779,
    2880746675, 1270133939, 2830415027, 1219802291, 2877600947, 1266988211, 2827269299, 1216656563,
    2881127603, 1270514867, 2830795955, 1220183219, 2877981875, 1267369139, 2827650227, 1217037491,
    2880734387, 1270121651, 2830402739, 1219790003, 2877588659, 1266975923, 2827257011, 1216644275,
    2881141379, 1270528643, 2830809731, 1220196995, 2877995651, 1267382915, 2827664003, 1217051267,
    2880748163, 1270135427, 2830416515, 1219803779, 2877602435, 1266989699, 2827270787, 1216658051,
    2881129091, 1270516355, 2830797443, 1220184707, 2877983363, 1267370627, 2827651715, 1217038979,
    2880735875, 1270123139, 2830404227, 1219791491, 2877590147, 1266977411, 2827258499, 1216645763,
    2881139843, 1270527107, 2830808195, 1220195459, 2877994115, 1267381379, 2827662467, 1217049731,
    2880746627, 1270133891, 2830414979, 1219802243, 2877600899, 1266988163, 2827269251, 1216656515,
    2881127555, 1270514819, 2830795907, 1220183171, 2877981827, 1267369091, 2827650179, 1217037443,
    2880734339, 1270121603, 2830402691, 1219789955, 2877588611, 1266975875, 2827256963, 1216644227,
    2965027518, 2159721150, 2961881790, 2156575422, 2964634302, 2159327934, 2961488574, 2156182206,
    2965015230, 2159708862, 2961869502, 2156563134, 2964622014, 2159315646, 2961476286, 2156169918,
    2965025982, 2159719614, 2961880254, 2156573886, 2964632766, 2159326398, 2961487038, 2156180670,
    2965013694, 2159707326, 2961867966, 2156561598, 2964620478, 2159314110, 2961474750, 2156168382,
    2965027470, 2159721102, 2961881742, 2156575374, 2964634254, 2159327886, 2961488526, 2156182158,
    2965015182, 2159708814, 2961869454, 2156563086, 2964621966, 2159315598, 2961476238, 2156169870,
    2965025934, 2159719566, 2961880206, 2156573838, 2964632718, 2159326350, 2961486990, 2156180622,
    2965013646, 2159707278, 2961867918, 2156561550, 2964620430, 2159314062, 2961474702, 2156168334,
    2965027507, 2159721139, 2961881779, 2156575411, 2964634291, 2159327923, 2961488563, 2156182195,
    2965015219, 2159708851, 2961869491, 2156563123, 2964622003, 2159315635, 2961476275, 2156169907,
    2965025971, 2159719603, 2961880243, 2156573875, 2964632755, 2159326387, 2961487027, 2156180659,
    2965013683, 2159707315, 2961867955, 2156561587, 2964620467, 2159314099, 2961474739, 2156168371,
    2965027459, 2159721091, 2961881731, 2156575363, 2964634243, 2159327875, 2961488515, 2156182147,
    2965015171, 2159708803, 2961869443, 2156563075, 2964621955, 2159315587, 2961476227, 2156169859,
    2965025923, 2159719555, 2961880195, 2156573827, 2964632707, 2159326339, 2961486979, 2156180611,
    2965013635, 2159707267, 2961867907, 2156561539, 2964620419, 2159314051, 2961474691, 2156168323,
    2976579765, 2171273397, 2976383157, 2171076789, 2976579717, 2171273349, 2976383109, 2171076741,
    3203381950, 2398075582, 3018832574, 2213526206, 3191847614, 2386541246, 3007298238, 2201991870,
    3203369662, 2398063294, 3018820286, 2213513918, 3191835326, 2386528958, 3007285950, 2201979582,
    3203380414, 2398074046, 3018831038, 2213524670, 3191846078, 2386539710, 3007296702, 2201990334,
    3203368126, 2398061758, 3018818750, 2213512382, 3191833790, 2386527422, 3007284414, 2201978046,
    3203381902, 2398075534, 3018832526, 2213526158, 3191847566, 2386541198, 3007298190, 2201991822,
    3203369614, 2398063246, 3018820238, 2213513870, 3191835278, 2386528910, 3007285902, 2201979534,
    3203380366, 2398073998, 3018830990, 2213524622, 3191846030, 2386539662, 3007296654, 2201990286,
    3203368078, 2398061710, 3018818702, 2213512334, 3191833742, 2386527374, 3007284366, 2201977998,
    3203381939, 2398075571, 3018832563, 2213526195, 3191847603, 2386541235, 3007298227, 2201991859,
    3203369651, 2398063283, 3018820275, 2213513907, 3191835315, 2386528947, 3007285939, 2201979571,
    3203380403, 2398074035, 3018831027, 2213524659, 3191846067, 2386539699, 3007296691, 2201990323,
    3203368115, 2398061747, 3018818739, 2213512371, 3191833779, 2386527411, 3007284403, 2201978035,
    3203381891, 2398075523, 3018832515, 2213526147, 3191847555, 2386541187, 3007298179, 2201991811,
    3203369603, 2398063235, 3018820227, 2213513859, 3191835267, 2386528899, 3007285891, 2201979523,
    3203380355, 2398073987, 3018830979, 2213524611, 3191846019, 2386539651, 3007296643, 2201990275,
    3203368067, 2398061699, 3018818691, 2213512323, 3191833731, 2386527363, 3007284355, 2201977987,
    3405691582, 3305028286, 3404970686, 3304307390, 3405679294, 3305015998, 3404958398, 3304295102,
    3405690046, 3305026750, 3404969150, 3304305854, 3405677758, 3305014462, 3404956862, 3304293566,
    3405691534, 3305028238, 3404970638, 3304307342, 3405679246, 3305015950, 3404958350, 3304295054,
    3405689998, 3305026702, 3404969102, 3304305806, 3405677710, 3305014414, 3404956814, 3304293518,
    3405691571, 3305028275, 3404970675, 3304307379, 3405679283, 3305015987, 3404958387, 3304295091,
    3405690035, 3305026739, 3404969139, 3304305843, 3405677747, 3305014451, 3404956851, 3304293555,
    3405691523, 3305028227, 3404970627, 3304307331, 3405679235, 3305015939, 3404958339, 3304295043,
    3405689987, 3305026691, 3404969091, 3304305795, 3405677699, 3305014403, 3404956803, 3304293507,
    3405697037, 3305033741, 3404976141, 3304312845, 3735927486, 3551378110, 3729636030, 3545086654,
    3735915198, 3551365822, 3729623742, 3545074366, 3735925950, 3551376574, 3729634494, 3545085118,
    3735913662, 3551364286, 3729622206, 3545072830, 3735927438, 3551378062, 3729635982, 3545086606,
    3735915150, 3551365774, 3729623694, 3545074318, 3735925902, 3551376526, 3729634446, 3545085070,
    3735913614, 3551364238, 3729622158, 3545072782, 3735927475, 3551378099, 3729636019, 3545086643,
    3735915187, 3551365811, 3729623731, 3545074355, 3735925939, 3551376563, 3729634483, 3545085107,
    3735913651, 3551364275, 3729622195, 3545072819, 3735927427, 3551378051, 3729635971, 3545086595,
    3735915139, 3551365763, 3729623683, 3545074307, 3735925891, 3551376515, 3729634435, 3545085059,
    3735913603, 3551364227, 3729622147, 3545072771, 3735932941, 3551383565, 3729641485, 3545092109,
    4027431614, 4027419326, 4027430078, 4027417790, 4027431566, 4027419278, 4027430030, 4027417742,
    4027431603, 4027419315, 4027430067, 4027417779, 4027431555, 4027419267, 4027430019, 4027417731,
    4276992702, 4092443326, 4265458366, 4080908990, 4276980414, 4092431038, 4265446078, 4080896702,
    4276991166, 4092441790, 4265456830, 4080907454, 4276978878, 4092429502, 4265444542, 4080895166,
    4276992654, 4092443278, 4265458318, 4080908942, 4276980366, 4092430990, 4265446030, 4080896654,
    4276991118, 4092441742, 4265456782, 4080907406, 4276978830, 4092429454, 4265444494, 4080895118,
    4276992691, 4092443315, 4265458355, 4080908979, 4276980403, 4092431027, 4265446067, 4080896691,
    4276991155, 4092441779, 4265456819, 4080907443, 4276978867, 4092429491, 4265444531, 4080895155,
    4276992643, 4092443267, 4265458307, 4080908931, 4276980355, 4092430979, 4265446019, 4080896643,
    4276991107, 4092441731, 4265456771, 4080907395, 4276978819, 4092429443, 4265444483, 4080895107,
    195934910, 145603262, 189643454, 139311806, 195922622, 145590974, 189631166, 139299518,
    195933374, 145601726, 189641918, 139310270, 195921086, 145589438, 189629630, 139297982,
    195934862, 145603214, 189643406, 139311758, 195922574, 145590926, 189631118, 139299470,
    195933326, 145601678, 189641870, 139310222, 195921038, 145589390, 189629582, 139297934,
    195934899, 145603251, 189643443, 139311795, 195922611, 145590963, 189631155, 139299507,
    195933363, 145601715, 189641907, 139310259, 195921075, 145589427, 189629619, 139297971,
    195934851, 145603203, 189643395, 139311747, 195922563, 145590915, 189631107, 139299459,
    195933315, 145601667, 189641859, 139310211, 195921027, 145589379, 189629571, 139297923,
    252707358, 252707347, 762133, 565525, 737557, 540949, 179681982, 79018686, 176536254, 75872958,
    179669694, 79006398, 176523966, 75860670, 179680446, 79017150, 176534718, 75871422, 179668158,
    79004862, 176522430, 75859134, 179681934, 79018638, 176536206, 75872910, 179669646, 79006350,
    176523918, 75860622, 179680398, 79017102, 176534670, 75871374, 179668110, 79004814, 176522382,
    75859086, 179681971, 79018675, 176536243, 75872947, 179669683, 79006387, 176523955, 75860659,
    179680435, 79017139, 176534707, 75871411, 179668147, 79004851, 176522419, 75859123, 179681923,
    79018627, 176536195, 75872899, 179669635, 79006339, 176523907, 75860611, 179680387, 79017091,
    176534659, 75871363, 179668099, 79004803, 176522371, 75859075, 173390526, 72727230, 173378238,
    72714942, 173388990, 72725694, 173376702, 72713406, 173390478, 72727182, 173378190, 72714894,
    173388942, 72725646, 173376654, 72713358, 173390515, 72727219, 173378227, 72714931, 173388979,
    72725683, 173376691, 72713395, 173390467, 72727171, 173378179, 72714883, 173388931, 72725635,
    173376643, 72713347,
];

const INTERNAL_COMPILER_DOCS_LINE: &str = "#### This error code is internal to the compiler and will not be emitted with normal Rust code.";

/// Parser states for `line_is_url`.
#[derive(Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
enum LIUState {
    EXP_COMMENT_START,
    EXP_LINK_LABEL_OR_URL,
    EXP_URL,
    EXP_END,
}

/// Returns `true` if `line` appears to be a line comment containing a URL,
/// possibly with a Markdown link label in front, and nothing else.
/// The Markdown link label, if present, may not contain whitespace.
/// Lines of this form are allowed to be overlength, because Markdown
/// offers no way to split a line in the middle of a URL, and the lengths
/// of URLs to external references are beyond our control.
fn line_is_url(is_error_code: bool, columns: usize, line: &str) -> bool {
    // more basic check for markdown, to avoid complexity in implementing two state machines
    if is_error_code {
        return line.starts_with('[') && line.contains("]:") && line.contains("http");
    }

    use self::LIUState::*;
    let mut state: LIUState = EXP_COMMENT_START;
    let is_url = |w: &str| w.starts_with("http://") || w.starts_with("https://");

    for tok in line.split_whitespace() {
        match (state, tok) {
            (EXP_COMMENT_START, "//") | (EXP_COMMENT_START, "///") | (EXP_COMMENT_START, "//!") => {
                state = EXP_LINK_LABEL_OR_URL
            }

            (EXP_LINK_LABEL_OR_URL, w)
                if w.len() >= 4 && w.starts_with('[') && w.ends_with("]:") =>
            {
                state = EXP_URL
            }

            (EXP_LINK_LABEL_OR_URL, w) if is_url(w) => state = EXP_END,

            (EXP_URL, w) if is_url(w) || w.starts_with("../") => state = EXP_END,

            (_, w) if w.len() > columns && is_url(w) => state = EXP_END,

            (_, _) => {}
        }
    }

    state == EXP_END
}

/// Returns `true` if `line` can be ignored. This is the case when it contains
/// an annotation that is explicitly ignored.
fn should_ignore(line: &str) -> bool {
    // Matches test annotations like `//~ ERROR text`.
    // This mirrors the regex in src/tools/compiletest/src/runtest.rs, please
    // update both if either are changed.
    lazy_static::lazy_static! {
        static ref ANNOTATION_RE: Regex = Regex::new("\\s*//(\\[.*\\])?~.*").unwrap();
    }
    // For `ui_test`-style UI test directives, also ignore
    // - `//@[rev] compile-flags`
    // - `//@[rev] normalize-stderr-test`
    lazy_static::lazy_static! {
        static ref UI_TEST_LONG_DIRECTIVES_RE: Regex =
        Regex::new("\\s*//@(\\[.*\\]) (compile-flags|normalize-stderr-test|error-pattern).*")
            .unwrap();
    }
    ANNOTATION_RE.is_match(line)
        || ANNOTATIONS_TO_IGNORE.iter().any(|a| line.contains(a))
        || UI_TEST_LONG_DIRECTIVES_RE.is_match(line)
}

/// Returns `true` if `line` is allowed to be longer than the normal limit.
fn long_line_is_ok(extension: &str, is_error_code: bool, max_columns: usize, line: &str) -> bool {
    match extension {
        // fluent files are allowed to be any length
        "ftl" => true,
        // non-error code markdown is allowed to be any length
        "md" if !is_error_code => true,
        // HACK(Ezrashaw): there is no way to split a markdown header over multiple lines
        "md" if line == INTERNAL_COMPILER_DOCS_LINE => true,
        _ => line_is_url(is_error_code, max_columns, line) || should_ignore(line),
    }
}

enum Directive {
    /// By default, tidy always warns against style issues.
    Deny,

    /// `Ignore(false)` means that an `ignore-tidy-*` directive
    /// has been provided, but is unnecessary. `Ignore(true)`
    /// means that it is necessary (i.e. a warning would be
    /// produced if `ignore-tidy-*` was not present).
    Ignore(bool),
}

fn contains_ignore_directive(can_contain: bool, contents: &str, check: &str) -> Directive {
    if !can_contain {
        return Directive::Deny;
    }
    // Update `can_contain` when changing this
    if contents.contains(&format!("// ignore-tidy-{check}"))
        || contents.contains(&format!("# ignore-tidy-{check}"))
        || contents.contains(&format!("/* ignore-tidy-{check} */"))
    {
        Directive::Ignore(false)
    } else {
        Directive::Deny
    }
}

macro_rules! suppressible_tidy_err {
    ($err:ident, $skip:ident, $msg:literal) => {
        if let Directive::Deny = $skip {
            $err(&format!($msg));
        } else {
            $skip = Directive::Ignore(true);
        }
    };
}

pub fn is_in(full_path: &Path, parent_folder_to_find: &str, folder_to_find: &str) -> bool {
    if let Some(parent) = full_path.parent() {
        if parent.file_name().map_or_else(
            || false,
            |f| {
                f.to_string_lossy() == folder_to_find
                    && parent
                        .parent()
                        .and_then(|f| f.file_name())
                        .map_or_else(|| false, |f| f == parent_folder_to_find)
            },
        ) {
            true
        } else {
            is_in(parent, parent_folder_to_find, folder_to_find)
        }
    } else {
        false
    }
}

fn skip_markdown_path(path: &Path) -> bool {
    // These aren't ready for tidy.
    const SKIP_MD: &[&str] = &[
        "src/doc/edition-guide",
        "src/doc/embedded-book",
        "src/doc/nomicon",
        "src/doc/reference",
        "src/doc/rust-by-example",
        "src/doc/rustc-dev-guide",
    ];
    SKIP_MD.iter().any(|p| path.ends_with(p))
}

fn is_unexplained_ignore(extension: &str, line: &str) -> bool {
    if !line.ends_with("```ignore") && !line.ends_with("```rust,ignore") {
        return false;
    }
    if extension == "md" && line.trim().starts_with("//") {
        // Markdown examples may include doc comments with ignore inside a
        // code block.
        return false;
    }
    true
}

pub fn check(path: &Path, bad: &mut bool) {
    fn skip(path: &Path, is_dir: bool) -> bool {
        if path.file_name().map_or(false, |name| name.to_string_lossy().starts_with(".#")) {
            // vim or emacs temporary file
            return true;
        }

        if filter_dirs(path) || skip_markdown_path(path) {
            return true;
        }

        // Don't check extensions for directories
        if is_dir {
            return false;
        }

        let extensions = ["rs", "py", "js", "sh", "c", "cpp", "h", "md", "css", "ftl", "goml"];

        // NB: don't skip paths without extensions (or else we'll skip all directories and will only check top level files)
        if path.extension().map_or(true, |ext| !extensions.iter().any(|e| ext == OsStr::new(e))) {
            return true;
        }

        // We only check CSS files in rustdoc.
        path.extension().map_or(false, |e| e == "css") && !is_in(path, "src", "librustdoc")
    }

    let problematic_consts_strings: Vec<String> = (PROBLEMATIC_CONSTS.iter().map(u32::to_string))
        .chain(PROBLEMATIC_CONSTS.iter().map(|v| format!("{:x}", v)))
        .chain(PROBLEMATIC_CONSTS.iter().map(|v| format!("{:X}", v)))
        .collect();
    let problematic_regex = RegexSet::new(problematic_consts_strings.as_slice()).unwrap();

    walk(path, skip, &mut |entry, contents| {
        let file = entry.path();
        let filename = file.file_name().unwrap().to_string_lossy();

        let is_style_file = filename.ends_with(".css");
        let under_rustfmt = filename.ends_with(".rs") &&
            // This list should ideally be sourced from rustfmt.toml but we don't want to add a toml
            // parser to tidy.
            !file.ancestors().any(|a| {
                (a.ends_with("tests") && a.join("COMPILER_TESTS.md").exists()) ||
                    a.ends_with("src/doc/book")
            });

        if contents.is_empty() {
            tidy_error!(bad, "{}: empty file", file.display());
        }

        let extension = file.extension().unwrap().to_string_lossy();
        let is_error_code = extension == "md" && is_in(file, "src", "error_codes");
        let is_goml_code = extension == "goml";

        let max_columns = if is_error_code {
            ERROR_CODE_COLS
        } else if is_goml_code {
            GOML_COLS
        } else {
            COLS
        };

        let can_contain = contents.contains("// ignore-tidy-")
            || contents.contains("# ignore-tidy-")
            || contents.contains("/* ignore-tidy-");
        // Enable testing ICE's that require specific (untidy)
        // file formats easily eg. `issue-1234-ignore-tidy.rs`
        if filename.contains("ignore-tidy") {
            return;
        }
        // Shell completions are automatically generated
        if let Some(p) = file.parent() {
            if p.ends_with(Path::new("src/etc/completions")) {
                return;
            }
        }
        let mut skip_cr = contains_ignore_directive(can_contain, &contents, "cr");
        let mut skip_undocumented_unsafe =
            contains_ignore_directive(can_contain, &contents, "undocumented-unsafe");
        let mut skip_tab = contains_ignore_directive(can_contain, &contents, "tab");
        let mut skip_line_length = contains_ignore_directive(can_contain, &contents, "linelength");
        let mut skip_file_length = contains_ignore_directive(can_contain, &contents, "filelength");
        let mut skip_end_whitespace =
            contains_ignore_directive(can_contain, &contents, "end-whitespace");
        let mut skip_trailing_newlines =
            contains_ignore_directive(can_contain, &contents, "trailing-newlines");
        let mut skip_leading_newlines =
            contains_ignore_directive(can_contain, &contents, "leading-newlines");
        let mut skip_copyright = contains_ignore_directive(can_contain, &contents, "copyright");
        let mut skip_dbg = contains_ignore_directive(can_contain, &contents, "dbg");
        let mut skip_odd_backticks =
            contains_ignore_directive(can_contain, &contents, "odd-backticks");
        let mut leading_new_lines = false;
        let mut trailing_new_lines = 0;
        let mut lines = 0;
        let mut last_safety_comment = false;
        let mut comment_block: Option<(usize, usize)> = None;
        let is_test = file.components().any(|c| c.as_os_str() == "tests");
        // scanning the whole file for multiple needles at once is more efficient than
        // executing lines times needles separate searches.
        let any_problematic_line = problematic_regex.is_match(contents);
        for (i, line) in contents.split('\n').enumerate() {
            if line.is_empty() {
                if i == 0 {
                    leading_new_lines = true;
                }
                trailing_new_lines += 1;
                continue;
            } else {
                trailing_new_lines = 0;
            }

            let trimmed = line.trim();

            if !trimmed.starts_with("//") {
                lines += 1;
            }

            let mut err = |msg: &str| {
                tidy_error!(bad, "{}:{}: {}", file.display(), i + 1, msg);
            };

            if trimmed.contains("dbg!")
                && !trimmed.starts_with("//")
                && !file.ancestors().any(|a| {
                    (a.ends_with("tests") && a.join("COMPILER_TESTS.md").exists())
                        || a.ends_with("library/alloc/tests")
                })
                && filename != "tests.rs"
            {
                suppressible_tidy_err!(
                    err,
                    skip_dbg,
                    "`dbg!` macro is intended as a debugging tool. It should not be in version control."
                )
            }

            if !under_rustfmt
                && line.chars().count() > max_columns
                && !long_line_is_ok(&extension, is_error_code, max_columns, line)
            {
                suppressible_tidy_err!(
                    err,
                    skip_line_length,
                    "line longer than {max_columns} chars"
                );
            }
            if !is_style_file && line.contains('\t') {
                suppressible_tidy_err!(err, skip_tab, "tab character");
            }
            if line.ends_with(' ') || line.ends_with('\t') {
                suppressible_tidy_err!(err, skip_end_whitespace, "trailing whitespace");
            }
            if is_style_file && line.starts_with(' ') {
                err("CSS files use tabs for indent");
            }
            if line.contains('\r') {
                suppressible_tidy_err!(err, skip_cr, "CR character");
            }
            if filename != "style.rs" {
                if trimmed.contains("TODO") {
                    err(
                        "TODO is used for tasks that should be done before merging a PR; If you want to leave a message in the codebase use FIXME",
                    )
                }
                if trimmed.contains("//") && trimmed.contains(" XXX") {
                    err("Instead of XXX use FIXME")
                }
                if any_problematic_line {
                    for s in problematic_consts_strings.iter() {
                        if trimmed.contains(s) {
                            err("Don't use magic numbers that spell things (consider 0x12345678)");
                        }
                    }
                }
            }
            // for now we just check libcore
            if trimmed.contains("unsafe {") && !trimmed.starts_with("//") && !last_safety_comment {
                if file.components().any(|c| c.as_os_str() == "core") && !is_test {
                    suppressible_tidy_err!(err, skip_undocumented_unsafe, "undocumented unsafe");
                }
            }
            if trimmed.contains("// SAFETY:") {
                last_safety_comment = true;
            } else if trimmed.starts_with("//") || trimmed.is_empty() {
                // keep previous value
            } else {
                last_safety_comment = false;
            }
            if (line.starts_with("// Copyright")
                || line.starts_with("# Copyright")
                || line.starts_with("Copyright"))
                && (trimmed.contains("Rust Developers")
                    || trimmed.contains("Rust Project Developers"))
            {
                suppressible_tidy_err!(
                    err,
                    skip_copyright,
                    "copyright notices attributed to the Rust Project Developers are deprecated"
                );
            }
            if !file.components().any(|c| c.as_os_str() == "rustc_baked_icu_data") {
                if is_unexplained_ignore(&extension, line) {
                    err(UNEXPLAINED_IGNORE_DOCTEST_INFO);
                }
            }

            if filename.ends_with(".cpp") && line.contains("llvm_unreachable") {
                err(LLVM_UNREACHABLE_INFO);
            }

            // For now only enforce in compiler
            let is_compiler = || file.components().any(|c| c.as_os_str() == "compiler");

            if is_compiler() {
                if line.contains("//")
                    && line
                        .chars()
                        .collect::<Vec<_>>()
                        .windows(4)
                        .any(|cs| matches!(cs, ['.', ' ', ' ', last] if last.is_alphabetic()))
                {
                    err(DOUBLE_SPACE_AFTER_DOT)
                }

                if filename.ends_with(".ftl") {
                    let line_backticks = trimmed.chars().filter(|ch| *ch == '`').count();
                    if line_backticks % 2 == 1 {
                        suppressible_tidy_err!(err, skip_odd_backticks, "odd number of backticks");
                    }
                } else if trimmed.contains("//") {
                    let (start_line, mut backtick_count) = comment_block.unwrap_or((i + 1, 0));
                    let line_backticks = trimmed.chars().filter(|ch| *ch == '`').count();
                    let comment_text = trimmed.split("//").nth(1).unwrap();
                    // This check ensures that we don't lint for code that has `//` in a string literal
                    if line_backticks % 2 == 1 {
                        backtick_count += comment_text.chars().filter(|ch| *ch == '`').count();
                    }
                    comment_block = Some((start_line, backtick_count));
                } else {
                    if let Some((start_line, backtick_count)) = comment_block.take() {
                        if backtick_count % 2 == 1 {
                            let mut err = |msg: &str| {
                                tidy_error!(bad, "{}:{start_line}: {msg}", file.display());
                            };
                            let block_len = (i + 1) - start_line;
                            if block_len == 1 {
                                suppressible_tidy_err!(
                                    err,
                                    skip_odd_backticks,
                                    "comment with odd number of backticks"
                                );
                            } else {
                                suppressible_tidy_err!(
                                    err,
                                    skip_odd_backticks,
                                    "{block_len}-line comment block with odd number of backticks"
                                );
                            }
                        }
                    }
                }
            }
        }
        if leading_new_lines {
            let mut err = |_| {
                tidy_error!(bad, "{}: leading newline", file.display());
            };
            suppressible_tidy_err!(err, skip_leading_newlines, "missing leading newline");
        }
        let mut err = |msg: &str| {
            tidy_error!(bad, "{}: {}", file.display(), msg);
        };
        match trailing_new_lines {
            0 => suppressible_tidy_err!(err, skip_trailing_newlines, "missing trailing newline"),
            1 => {}
            n => suppressible_tidy_err!(
                err,
                skip_trailing_newlines,
                "too many trailing newlines ({n})"
            ),
        };
        if lines > LINES {
            let mut err = |_| {
                tidy_error!(
                    bad,
                    "{}: too many lines ({}) (add `// \
                     ignore-tidy-filelength` to the file to suppress this error)",
                    file.display(),
                    lines
                );
            };
            suppressible_tidy_err!(err, skip_file_length, "");
        }

        if let Directive::Ignore(false) = skip_cr {
            tidy_error!(bad, "{}: ignoring CR characters unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_tab {
            tidy_error!(bad, "{}: ignoring tab characters unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_end_whitespace {
            tidy_error!(bad, "{}: ignoring trailing whitespace unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_trailing_newlines {
            tidy_error!(bad, "{}: ignoring trailing newlines unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_leading_newlines {
            tidy_error!(bad, "{}: ignoring leading newlines unnecessarily", file.display());
        }
        if let Directive::Ignore(false) = skip_copyright {
            tidy_error!(bad, "{}: ignoring copyright unnecessarily", file.display());
        }
        // We deliberately do not warn about these being unnecessary,
        // that would just lead to annoying churn.
        let _unused = skip_line_length;
        let _unused = skip_file_length;
    })
}
