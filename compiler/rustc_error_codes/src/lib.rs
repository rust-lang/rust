//! This library is used to gather all error codes into one place, to make
//! their maintenance easier.

// tidy-alphabetical-start
#![deny(rustdoc::invalid_codeblock_attributes)]
// tidy-alphabetical-end

// This higher-order macro defines the error codes that are in use. It is used
// in the `rustc_errors` crate. Removed error codes are listed in the comment
// below.
//
// /!\ IMPORTANT /!\
//
// Error code explanation are defined in `error_codes/EXXXX.md` files. They must follow the RFC
// 1567 available here:
// https://rust-lang.github.io/rfcs/1567-long-error-codes-explanation-normalization.html
//
// Also, the contents of this macro is checked by tidy (in `check_error_codes_docs`). If you change
// the macro syntax you will need to change tidy as well.
//
// Do *not* remove entries from this list. Instead, just add a note to the corresponding markdown
// file saying that this error is not emitted by the compiler any more (see E0001.md for an
// example), and remove all code examples that do not build any more.
#[macro_export]
#[rustfmt::skip]
macro_rules! error_codes {
    ($macro:path) => (
        $macro!(
0001,
0002,
0004,
0005,
0007,
0009,
0010,
0013,
0014,
0015,
0023,
0025,
0026,
0027,
0029,
0030,
0033,
0034,
0038,
0040,
0044,
0045,
0046,
0049,
0050,
0053,
0054,
0055,
0057,
0059,
0060,
0061,
0062,
0063,
0067,
0069,
0070,
0071,
0072,
0073,
0074,
0075,
0076,
0077,
0080,
0081,
0084,
0087,
0088,
0089,
0090,
0091,
0092,
0093,
0094,
0106,
0107,
0109,
0110,
0116,
0117,
0118,
0119,
0120,
0121,
0124,
0128,
0130,
0131,
0132,
0133,
0136,
0137,
0138,
0139,
0152,
0154,
0158,
0161,
0162,
0164,
0165,
0170,
0178,
0183,
0184,
0185,
0186,
0191,
0192,
0193,
0195,
0197,
0198,
0199,
0200,
0201,
0203,
0204,
0205,
0206,
0207,
0208,
0210,
0211,
0212,
0214,
0220,
0221,
0222,
0223,
0224,
0225,
0226,
0227,
0228,
0229,
0230,
0231,
0232,
0243,
0244,
0251,
0252,
0253,
0254,
0255,
0256,
0259,
0260,
0261,
0262,
0263,
0264,
0267,
0268,
0271,
0275,
0276,
0277,
0281,
0282,
0283,
0284,
0297,
0301,
0302,
0303,
0307,
0308,
0309,
0310,
0311,
0312,
0316,
0317,
0320,
0321,
0322,
0323,
0324,
0325,
0326,
0328,
0329,
0364,
0365,
0366,
0367,
0368,
0369,
0370,
0371,
0373,
0374,
0375,
0376,
0377,
0378,
0379,
0380,
0381,
0382,
0383,
0384,
0386,
0387,
0388,
0389,
0390,
0391,
0392,
0393,
0398,
0399,
0401,
0403,
0404,
0405,
0407,
0408,
0409,
0411,
0412,
0415,
0416,
0422,
0423,
0424,
0425,
0426,
0428,
0429,
0430,
0431,
0432,
0433,
0434,
0435,
0436,
0437,
0438,
0439,
0445,
0446,
0447,
0448,
0449,
0451,
0452,
0453,
0454,
0455,
0457,
0458,
0459,
0460,
0461,
0462,
0463,
0464,
0466,
0468,
0469,
0472,
0476,
0477,
0478,
0482,
0491,
0492,
0493,
0495,
0496,
0497,
0498,
0499,
0500,
0501,
0502,
0503,
0504,
0505,
0506,
0507,
0508,
0509,
0510,
0511,
0512,
0514,
0515,
0516,
0517,
0518,
0519,
0520,
0521,
0522,
0523,
0524,
0525,
0527,
0528,
0529,
0530,
0531,
0532,
0533,
0534,
0535,
0536,
0537,
0538,
0539,
0541,
0542,
0543,
0544,
0545,
0546,
0547,
0549,
0550,
0551,
0552,
0554,
0556,
0557,
0559,
0560,
0561,
0562,
0565,
0566,
0567,
0568,
0569,
0570,
0571,
0572,
0573,
0574,
0575,
0576,
0577,
0578,
0579,
0580,
0581,
0582,
0583,
0584,
0585,
0586,
0587,
0588,
0589,
0590,
0591,
0592,
0593,
0594,
0595,
0596,
0597,
0599,
0600,
0601,
0602,
0603,
0604,
0605,
0606,
0607,
0608,
0609,
0610,
0614,
0615,
0616,
0617,
0618,
0619,
0620,
0621,
0622, // REMOVED: rustc-intrinsic ABI was removed
0623,
0624,
0625,
0626,
0627,
0628,
0631,
0632,
0633,
0634,
0635,
0636,
0637,
0638,
0639,
0640,
0641,
0642,
0643,
0644,
0646,
0647,
0648,
0657,
0658,
0659,
0660,
0661,
0662,
0663,
0664,
0665,
0666,
0667,
0668,
0669,
0670,
0671,
0687,
0688,
0689,
0690,
0691,
0692,
0693,
0695,
0696,
0697,
0698,
0699, // REMOVED: merged into generic inference var error
0700,
0701,
0703,
0704,
0705,
0706,
0708,
0710,
0712,
0713,
0714,
0715,
0716,
0711,
0717,
0718,
0719,
0720,
0722,
0724,
0725,
0726,
0727,
0728,
0729,
0730,
0731,
0732,
0733,
0734,
0735,
0736,
0737,
0739,
0740,
0741,
0742,
0743,
0744,
0745,
0746,
0747,
0748,
0749,
0750,
0751,
0752,
0753,
0754,
0755,
0756,
0757,
0758,
0759,
0760,
0761,
0762,
0763,
0764,
0765,
0766,
0767,
0768,
0769,
0770,
0771,
0772,
0773, // REMOVED: no longer an error
0774,
0775,
0776,
0777,
0778,
0779,
0780,
0781,
0782,
0783,
0784,
0785,
0786,
0787,
0788,
0789,
0790,
0791,
0792,
0793,
0794,
0795,
0796,
0797,
0798,
0799,
0800,
0801,
0802,
0803,
0804,
0805,
        );
    )
}

// Undocumented removed error codes. Note that many removed error codes are kept in the list above
// and marked as no-longer emitted with a note in the markdown file (see E0001 for an example).
//  E0006, // merged with E0005
//  E0008, // cannot bind by-move into a pattern guard
//  E0019, // merged into E0015
//  E0035, // merged into E0087/E0089
//  E0036, // merged into E0087/E0089
//  E0068,
//  E0085,
//  E0086,
//  E0101, // replaced with E0282
//  E0102, // replaced with E0282
//  E0103,
//  E0104,
//  E0122, // bounds in type aliases are ignored, turned into proper lint
//  E0123,
//  E0127,
//  E0129,
//  E0134,
//  E0135,
//  E0141,
//  E0153, // unused error code
//  E0157, // unused error code
//  E0159, // use of trait `{}` as struct constructor
//  E0163, // merged into E0071
//  E0167,
//  E0168,
//  E0172, // non-trait found in a type sum, moved to resolve
//  E0173, // manual implementations of unboxed closure traits are experimental
//  E0174,
//  E0182, // merged into E0229
//  E0187, // cannot infer the kind of the closure
//  E0188, // can not cast an immutable reference to a mutable pointer
//  E0189, // deprecated: can only cast a boxed pointer to a boxed object
//  E0190, // deprecated: can only cast a &-pointer to an &-object
//  E0194, // merged into E0403
//  E0196, // cannot determine a type for this closure
//  E0209, // builtin traits can only be implemented on structs or enums
//  E0213, // associated types are not accepted in this context
//  E0215, // angle-bracket notation is not stable with `Fn`
//  E0216, // parenthetical notation is only stable with `Fn`
//  E0217, // ambiguous associated type, defined in multiple supertraits
//  E0218, // no associated type defined
//  E0219, // associated type defined in higher-ranked supertrait
//  E0233,
//  E0234,
//  E0235, // structure constructor specifies a structure of type but
//  E0236, // no lang item for range syntax
//  E0237, // no lang item for range syntax
//  E0238, // parenthesized parameters may only be used with a trait
//  E0239, // `next` method of `Iterator` trait has unexpected type
//  E0240,
//  E0241,
//  E0242,
//  E0245, // not a trait
//  E0246, // invalid recursive type
//  E0247,
//  E0248, // value used as a type, now reported earlier during resolution
//         // as E0412
//  E0249,
//  E0257,
//  E0258,
//  E0272, // on_unimplemented #0
//  E0273, // on_unimplemented #1
//  E0274, // on_unimplemented #2
//  E0278, // requirement is not satisfied
//  E0279,
//  E0280, // changed to ICE
//  E0285, // overflow evaluation builtin bounds
//  E0296, // replaced with a generic attribute input check
//  E0298, // cannot compare constants
//  E0299, // mismatched types between arms
//  E0300, // unexpanded macro
//  E0304, // expected signed integer constant
//  E0305, // expected constant
//  E0313, // removed: found unreachable
//  E0314, // closure outlives stack frame
//  E0315, // cannot invoke closure outside of its lifetime
//  E0319, // trait impls for defaulted traits allowed just for structs/enums
//  E0372, // coherence not dyn-compatible
//  E0385, // {} in an aliasable location
//  E0402, // cannot use an outer type parameter in this context
//  E0406, // merged into 420
//  E0410, // merged into 408
//  E0413, // merged into 530
//  E0414, // merged into 530
//  E0417, // merged into 532
//  E0418, // merged into 532
//  E0419, // merged into 531
//  E0420, // merged into 532
//  E0421, // merged into 531
//  E0427, // merged into 530
//  E0445, // merged into 446 and type privacy lints
//  E0456, // plugin `..` is not available for triple `..`
//  E0465, // removed: merged with E0464
//  E0467, // removed
//  E0470, // removed
//  E0471, // constant evaluation error (in pattern)
//  E0473, // dereference of reference outside its lifetime
//  E0474, // captured variable `..` does not outlive the enclosing closure
//  E0475, // index of slice outside its lifetime
//  E0479, // the type `..` (provided as the value of a type parameter) is...
//  E0480, // lifetime of method receiver does not outlive the method call
//  E0481, // lifetime of function argument does not outlive the function call
//  E0483, // lifetime of operand does not outlive the operation
//  E0484, // reference is not valid at the time of borrow
//  E0485, // automatically reference is not valid at the time of borrow
//  E0486, // type of expression contains references that are not valid during..
//  E0487, // unsafe use of destructor: destructor might be called while...
//  E0488, // lifetime of variable does not enclose its declaration
//  E0489, // type/lifetime parameter not in scope here
//  E0490, // removed: unreachable
//  E0526, // shuffle indices are not constant
//  E0540, // multiple rustc_deprecated attributes
//  E0548, // replaced with a generic attribute input check
//  E0553, // multiple rustc_const_unstable attributes
//  E0555, // replaced with a generic attribute input check
//  E0558, // replaced with a generic attribute input check
//  E0563, // cannot determine a type for this `impl Trait` removed in 6383de15
//  E0564, // only named lifetimes are allowed in `impl Trait`,
//         // but `{}` was found in the type `{}`
//  E0598, // lifetime of {} is too short to guarantee its contents can be...
//  E0611, // merged into E0616
//  E0612, // merged into E0609
//  E0613, // Removed (merged with E0609)
//  E0629, // missing 'feature' (rustc_const_unstable)
//  E0630, // rustc_const_unstable attribute must be paired with stable/unstable
//         // attribute
//  E0645, // trait aliases not finished
//  E0694, // an unknown tool name found in scoped attributes
//  E0702, // replaced with a generic attribute input check
//  E0707, // multiple elided lifetimes used in arguments of `async fn`
//  E0709, // multiple different lifetimes used in arguments of `async fn`
//  E0721, // `await` keyword
//  E0722, // replaced with a generic attribute input check
//  E0723, // unstable feature in `const` context
//  E0738, // Removed; errored on `#[track_caller] fn`s in `extern "Rust" { ... }`.
//  E0744, // merged into E0728
//  E0776, // Removed; `#[cmse_nonsecure_entry]` is now `extern "cmse-nonsecure-entry"`
//  E0796, // unused error code. We use `static_mut_refs` lint instead.
