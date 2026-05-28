fn main() {
    0b121; //~ ERROR invalid digit for a base 2 literal
    0b10_10301; //~ ERROR invalid digit for a base 2 literal
    0b30; //~ ERROR invalid digit for a base 2 literal
    0b41; //~ ERROR invalid digit for a base 2 literal
    0b5; //~ ERROR invalid digit for a base 2 literal
    0b6; //~ ERROR invalid digit for a base 2 literal
    0b7; //~ ERROR invalid digit for a base 2 literal
    0b8; //~ ERROR invalid digit for a base 2 literal
    0b9; //~ ERROR invalid digit for a base 2 literal
}
