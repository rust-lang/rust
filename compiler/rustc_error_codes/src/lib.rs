//! This library is used to gather all error codes into one place, to make
//! their maintenance easier.

// tidy-alphabetical-start
#![allow(internal_features)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![doc(rust_logo)]
#![feature(rustdoc_internals)]
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
// Both columns are necessary because it's not possible in Rust to create a new identifier such as
// `E0123` from an integer literal such as `0123`, unfortunately.
//
// Do *not* remove entries from this list. Instead, just add a note th the corresponding markdown
// file saying that this error is not emitted by the compiler any more (see E0001.md for an
// example), and remove all code examples that do not build any more.
#[macro_export]
macro_rules! error_codes {
    ($macro:path) => (
        $macro!(
E0001: 0001,
E0002: 0002,
E0004: 0004,
E0005: 0005,
E0007: 0007,
E0009: 0009,
E0010: 0010,
E0013: 0013,
E0014: 0014,
E0015: 0015,
E0023: 0023,
E0025: 0025,
E0026: 0026,
E0027: 0027,
E0029: 0029,
E0030: 0030,
E0033: 0033,
E0034: 0034,
E0038: 0038,
E0040: 0040,
E0044: 0044,
E0045: 0045,
E0046: 0046,
E0049: 0049,
E0050: 0050,
E0053: 0053,
E0054: 0054,
E0055: 0055,
E0057: 0057,
E0059: 0059,
E0060: 0060,
E0061: 0061,
E0062: 0062,
E0063: 0063,
E0067: 0067,
E0069: 0069,
E0070: 0070,
E0071: 0071,
E0072: 0072,
E0073: 0073,
E0074: 0074,
E0075: 0075,
E0076: 0076,
E0077: 0077,
E0080: 0080,
E0081: 0081,
E0084: 0084,
E0087: 0087,
E0088: 0088,
E0089: 0089,
E0090: 0090,
E0091: 0091,
E0092: 0092,
E0093: 0093,
E0094: 0094,
E0106: 0106,
E0107: 0107,
E0109: 0109,
E0110: 0110,
E0116: 0116,
E0117: 0117,
E0118: 0118,
E0119: 0119,
E0120: 0120,
E0121: 0121,
E0124: 0124,
E0128: 0128,
E0130: 0130,
E0131: 0131,
E0132: 0132,
E0133: 0133,
E0136: 0136,
E0137: 0137,
E0138: 0138,
E0139: 0139,
E0152: 0152,
E0154: 0154,
E0158: 0158,
E0161: 0161,
E0162: 0162,
E0164: 0164,
E0165: 0165,
E0170: 0170,
E0178: 0178,
E0183: 0183,
E0184: 0184,
E0185: 0185,
E0186: 0186,
E0191: 0191,
E0192: 0192,
E0193: 0193,
E0195: 0195,
E0197: 0197,
E0198: 0198,
E0199: 0199,
E0200: 0200,
E0201: 0201,
E0203: 0203,
E0204: 0204,
E0205: 0205,
E0206: 0206,
E0207: 0207,
E0208: 0208,
E0210: 0210,
E0211: 0211,
E0212: 0212,
E0214: 0214,
E0220: 0220,
E0221: 0221,
E0222: 0222,
E0223: 0223,
E0224: 0224,
E0225: 0225,
E0226: 0226,
E0227: 0227,
E0228: 0228,
E0229: 0229,
E0230: 0230,
E0231: 0231,
E0232: 0232,
E0243: 0243,
E0244: 0244,
E0251: 0251,
E0252: 0252,
E0253: 0253,
E0254: 0254,
E0255: 0255,
E0256: 0256,
E0259: 0259,
E0260: 0260,
E0261: 0261,
E0262: 0262,
E0263: 0263,
E0264: 0264,
E0267: 0267,
E0268: 0268,
E0271: 0271,
E0275: 0275,
E0276: 0276,
E0277: 0277,
E0281: 0281,
E0282: 0282,
E0283: 0283,
E0284: 0284,
E0297: 0297,
E0301: 0301,
E0302: 0302,
E0303: 0303,
E0307: 0307,
E0308: 0308,
E0309: 0309,
E0310: 0310,
E0311: 0311,
E0312: 0312,
E0316: 0316,
E0317: 0317,
E0320: 0320,
E0321: 0321,
E0322: 0322,
E0323: 0323,
E0324: 0324,
E0325: 0325,
E0326: 0326,
E0328: 0328,
E0329: 0329,
E0364: 0364,
E0365: 0365,
E0366: 0366,
E0367: 0367,
E0368: 0368,
E0369: 0369,
E0370: 0370,
E0371: 0371,
E0373: 0373,
E0374: 0374,
E0375: 0375,
E0376: 0376,
E0377: 0377,
E0378: 0378,
E0379: 0379,
E0380: 0380,
E0381: 0381,
E0382: 0382,
E0383: 0383,
E0384: 0384,
E0386: 0386,
E0387: 0387,
E0388: 0388,
E0389: 0389,
E0390: 0390,
E0391: 0391,
E0392: 0392,
E0393: 0393,
E0398: 0398,
E0399: 0399,
E0401: 0401,
E0403: 0403,
E0404: 0404,
E0405: 0405,
E0407: 0407,
E0408: 0408,
E0409: 0409,
E0411: 0411,
E0412: 0412,
E0415: 0415,
E0416: 0416,
E0422: 0422,
E0423: 0423,
E0424: 0424,
E0425: 0425,
E0426: 0426,
E0428: 0428,
E0429: 0429,
E0430: 0430,
E0431: 0431,
E0432: 0432,
E0433: 0433,
E0434: 0434,
E0435: 0435,
E0436: 0436,
E0437: 0437,
E0438: 0438,
E0439: 0439,
E0445: 0445,
E0446: 0446,
E0447: 0447,
E0448: 0448,
E0449: 0449,
E0451: 0451,
E0452: 0452,
E0453: 0453,
E0454: 0454,
E0455: 0455,
E0457: 0457,
E0458: 0458,
E0459: 0459,
E0460: 0460,
E0461: 0461,
E0462: 0462,
E0463: 0463,
E0464: 0464,
E0466: 0466,
E0468: 0468,
E0469: 0469,
E0472: 0472,
E0476: 0476,
E0477: 0477,
E0478: 0478,
E0482: 0482,
E0491: 0491,
E0492: 0492,
E0493: 0493,
E0495: 0495,
E0496: 0496,
E0497: 0497,
E0498: 0498,
E0499: 0499,
E0500: 0500,
E0501: 0501,
E0502: 0502,
E0503: 0503,
E0504: 0504,
E0505: 0505,
E0506: 0506,
E0507: 0507,
E0508: 0508,
E0509: 0509,
E0510: 0510,
E0511: 0511,
E0512: 0512,
E0514: 0514,
E0515: 0515,
E0516: 0516,
E0517: 0517,
E0518: 0518,
E0519: 0519,
E0520: 0520,
E0521: 0521,
E0522: 0522,
E0523: 0523,
E0524: 0524,
E0525: 0525,
E0527: 0527,
E0528: 0528,
E0529: 0529,
E0530: 0530,
E0531: 0531,
E0532: 0532,
E0533: 0533,
E0534: 0534,
E0535: 0535,
E0536: 0536,
E0537: 0537,
E0538: 0538,
E0539: 0539,
E0541: 0541,
E0542: 0542,
E0543: 0543,
E0544: 0544,
E0545: 0545,
E0546: 0546,
E0547: 0547,
E0549: 0549,
E0550: 0550,
E0551: 0551,
E0552: 0552,
E0554: 0554,
E0556: 0556,
E0557: 0557,
E0559: 0559,
E0560: 0560,
E0561: 0561,
E0562: 0562,
E0565: 0565,
E0566: 0566,
E0567: 0567,
E0568: 0568,
E0569: 0569,
E0570: 0570,
E0571: 0571,
E0572: 0572,
E0573: 0573,
E0574: 0574,
E0575: 0575,
E0576: 0576,
E0577: 0577,
E0578: 0578,
E0579: 0579,
E0580: 0580,
E0581: 0581,
E0582: 0582,
E0583: 0583,
E0584: 0584,
E0585: 0585,
E0586: 0586,
E0587: 0587,
E0588: 0588,
E0589: 0589,
E0590: 0590,
E0591: 0591,
E0592: 0592,
E0593: 0593,
E0594: 0594,
E0595: 0595,
E0596: 0596,
E0597: 0597,
E0599: 0599,
E0600: 0600,
E0601: 0601,
E0602: 0602,
E0603: 0603,
E0604: 0604,
E0605: 0605,
E0606: 0606,
E0607: 0607,
E0608: 0608,
E0609: 0609,
E0610: 0610,
E0614: 0614,
E0615: 0615,
E0616: 0616,
E0617: 0617,
E0618: 0618,
E0619: 0619,
E0620: 0620,
E0621: 0621,
E0622: 0622, // REMOVED: rustc-intrinsic ABI was removed
E0623: 0623,
E0624: 0624,
E0625: 0625,
E0626: 0626,
E0627: 0627,
E0628: 0628,
E0631: 0631,
E0632: 0632,
E0633: 0633,
E0634: 0634,
E0635: 0635,
E0636: 0636,
E0637: 0637,
E0638: 0638,
E0639: 0639,
E0640: 0640,
E0641: 0641,
E0642: 0642,
E0643: 0643,
E0644: 0644,
E0646: 0646,
E0647: 0647,
E0648: 0648,
E0657: 0657,
E0658: 0658,
E0659: 0659,
E0660: 0660,
E0661: 0661,
E0662: 0662,
E0663: 0663,
E0664: 0664,
E0665: 0665,
E0666: 0666,
E0667: 0667,
E0668: 0668,
E0669: 0669,
E0670: 0670,
E0671: 0671,
E0687: 0687,
E0688: 0688,
E0689: 0689,
E0690: 0690,
E0691: 0691,
E0692: 0692,
E0693: 0693,
E0695: 0695,
E0696: 0696,
E0697: 0697,
E0698: 0698,
E0699: 0699, // REMOVED: merged into generic inference var error
E0700: 0700,
E0701: 0701,
E0703: 0703,
E0704: 0704,
E0705: 0705,
E0706: 0706,
E0708: 0708,
E0710: 0710,
E0712: 0712,
E0713: 0713,
E0714: 0714,
E0715: 0715,
E0716: 0716,
E0711: 0711,
E0717: 0717,
E0718: 0718,
E0719: 0719,
E0720: 0720,
E0722: 0722,
E0724: 0724,
E0725: 0725,
E0726: 0726,
E0727: 0727,
E0728: 0728,
E0729: 0729,
E0730: 0730,
E0731: 0731,
E0732: 0732,
E0733: 0733,
E0734: 0734,
E0735: 0735,
E0736: 0736,
E0737: 0737,
E0739: 0739,
E0740: 0740,
E0741: 0741,
E0742: 0742,
E0743: 0743,
E0744: 0744,
E0745: 0745,
E0746: 0746,
E0747: 0747,
E0748: 0748,
E0749: 0749,
E0750: 0750,
E0751: 0751,
E0752: 0752,
E0753: 0753,
E0754: 0754,
E0755: 0755,
E0756: 0756,
E0757: 0757,
E0758: 0758,
E0759: 0759,
E0760: 0760,
E0761: 0761,
E0762: 0762,
E0763: 0763,
E0764: 0764,
E0765: 0765,
E0766: 0766,
E0767: 0767,
E0768: 0768,
E0769: 0769,
E0770: 0770,
E0771: 0771,
E0772: 0772,
E0773: 0773,
E0774: 0774,
E0775: 0775,
E0776: 0776,
E0777: 0777,
E0778: 0778,
E0779: 0779,
E0780: 0780,
E0781: 0781,
E0782: 0782,
E0783: 0783,
E0784: 0784,
E0785: 0785,
E0786: 0786,
E0787: 0787,
E0788: 0788,
E0789: 0789,
E0790: 0790,
E0791: 0791,
E0792: 0792,
E0793: 0793,
E0794: 0794,
E0795: 0795,
E0796: 0796,
E0797: 0797,
E0798: 0798,
E0799: 0799,
E0800: 0800,
E0801: 0801,
E0802: 0802,
E0803: 0803,
E0804: 0804,
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
//  E0723, // unstable feature in `const` context
//  E0738, // Removed; errored on `#[track_caller] fn`s in `extern "Rust" { ... }`.
//  E0744, // merged into E0728
//  E0776, // Removed; cmse_nonsecure_entry is now `C-cmse-nonsecure-entry`
//  E0796, // unused error code. We use `static_mut_refs` lint instead.
