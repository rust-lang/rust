enum Enum {
    //~^ ERROR discriminant value `3` assigned more than once
    P = 3,
    //~^ NOTE `3` assigned here
    X = 3,
    //~^ NOTE `3` assigned here
    Y = 5
}

#[repr(u8)]
enum EnumOverflowRepr {
    //~^ ERROR discriminant value `1` assigned more than once
    P = 257,
    //~^ NOTE `1` (overflowed from `257`) assigned here
    X = 513,
    //~^ NOTE `1` (overflowed from `513`) assigned here
}

#[repr(i8)]
enum NegDisEnum {
    //~^ ERROR discriminant value `-1` assigned more than once
    First = -1,
    //~^ NOTE `-1` assigned here
    Second = -2,
    //~^ NOTE discriminant for `Last` incremented from this startpoint (`Second` + 1 variant later => `Last` = -1)
    Last,
    //~^ NOTE `-1` assigned here
}

enum MultipleDuplicates {
    //~^ ERROR discriminant value `0` assigned more than once
    //~^^ ERROR discriminant value `-2` assigned more than once
    V0,
    //~^ NOTE `0` assigned here
    V1 = 0,
    //~^ NOTE `0` assigned here
    V2,
    V3,
    V4 = 0,
    //~^ NOTE `0` assigned here
    V5 = -2,
    //~^ NOTE discriminant for `V7` incremented from this startpoint (`V5` + 2 variants later => `V7` = 0)
    //~^^ NOTE `-2` assigned here
    V6,
    V7,
    //~^ NOTE `0` assigned here
    V8 = -3,
    //~^ NOTE discriminant for `V9` incremented from this startpoint (`V8` + 1 variant later => `V9` = -2)
    V9,
    //~^ NOTE `-2` assigned here
}

// Test for #131902
// Ensure that casting an enum with too many variants for its repr
// does not ICE
#[repr(u8)]
enum TooManyVariants {
    //~^ ERROR discriminant value `0` assigned more than once
    X000, X001, X002, X003, X004, X005, X006, X007, X008, X009,
    //~^ NOTE `0` assigned here
    //~| NOTE discriminant for `X256` incremented from this startpoint
    X010, X011, X012, X013, X014, X015, X016, X017, X018, X019,
    X020, X021, X022, X023, X024, X025, X026, X027, X028, X029,
    X030, X031, X032, X033, X034, X035, X036, X037, X038, X039,
    X040, X041, X042, X043, X044, X045, X046, X047, X048, X049,
    X050, X051, X052, X053, X054, X055, X056, X057, X058, X059,
    X060, X061, X062, X063, X064, X065, X066, X067, X068, X069,
    X070, X071, X072, X073, X074, X075, X076, X077, X078, X079,
    X080, X081, X082, X083, X084, X085, X086, X087, X088, X089,
    X090, X091, X092, X093, X094, X095, X096, X097, X098, X099,
    X100, X101, X102, X103, X104, X105, X106, X107, X108, X109,
    X110, X111, X112, X113, X114, X115, X116, X117, X118, X119,
    X120, X121, X122, X123, X124, X125, X126, X127, X128, X129,
    X130, X131, X132, X133, X134, X135, X136, X137, X138, X139,
    X140, X141, X142, X143, X144, X145, X146, X147, X148, X149,
    X150, X151, X152, X153, X154, X155, X156, X157, X158, X159,
    X160, X161, X162, X163, X164, X165, X166, X167, X168, X169,
    X170, X171, X172, X173, X174, X175, X176, X177, X178, X179,
    X180, X181, X182, X183, X184, X185, X186, X187, X188, X189,
    X190, X191, X192, X193, X194, X195, X196, X197, X198, X199,
    X200, X201, X202, X203, X204, X205, X206, X207, X208, X209,
    X210, X211, X212, X213, X214, X215, X216, X217, X218, X219,
    X220, X221, X222, X223, X224, X225, X226, X227, X228, X229,
    X230, X231, X232, X233, X234, X235, X236, X237, X238, X239,
    X240, X241, X242, X243, X244, X245, X246, X247, X248, X249,
    X250, X251, X252, X253, X254, X255,
    X256,
    //~^ ERROR enum discriminant overflowed
    //~| NOTE overflowed on value after 255
    //~| NOTE explicitly set `X256 = 0`
    //~| NOTE `0` assigned here
}

fn main() {
    TooManyVariants::X256 as u8;
}
