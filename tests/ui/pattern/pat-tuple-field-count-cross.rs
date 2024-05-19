//@ aux-build:declarations-for-tuple-field-count-errors.rs

extern crate declarations_for_tuple_field_count_errors;

use declarations_for_tuple_field_count_errors::*;

fn main() {
    match Z0 {
        Z0() => {} //~ ERROR expected tuple struct or tuple variant, found unit struct `Z0`
        Z0(x) => {} //~ ERROR expected tuple struct or tuple variant, found unit struct `Z0`
    }
    match Z1() {
        Z1 => {} //~ ERROR match bindings cannot shadow tuple structs
        Z1(x) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple struct has 0 fields
    }

    match S(1, 2, 3) {
        S() => {} //~ ERROR this pattern has 0 fields, but the corresponding tuple struct has 3 fields
        S(1) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple struct has 3 fields
        S(xyz, abc) => {} //~ ERROR this pattern has 2 fields, but the corresponding tuple struct has 3 fields
        S(1, 2, 3, 4) => {} //~ ERROR this pattern has 4 fields, but the corresponding tuple struct has 3 fields
    }
    match M(1, 2, 3) {
        M() => {} //~ ERROR this pattern has 0 fields, but the corresponding tuple struct has 3 fields
        M(1) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple struct has 3 fields
        M(xyz, abc) => {} //~ ERROR this pattern has 2 fields, but the corresponding tuple struct has 3 fields
        M(1, 2, 3, 4) => {} //~ ERROR this pattern has 4 fields, but the corresponding tuple struct has 3 fields
    }

    match E1::Z0 {
        E1::Z0() => {} //~ ERROR expected tuple struct or tuple variant, found unit variant `E1::Z0`
        E1::Z0(x) => {} //~ ERROR expected tuple struct or tuple variant, found unit variant `E1::Z0`
    }
    match E1::Z1() {
        E1::Z1 => {} //~ ERROR expected unit struct, unit variant or constant, found tuple variant `E1::Z1`
        E1::Z1(x) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple variant has 0 fields
    }
    match E1::S(1, 2, 3) {
        E1::S() => {} //~ ERROR this pattern has 0 fields, but the corresponding tuple variant has 3 fields
        E1::S(1) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple variant has 3 fields
        E1::S(xyz, abc) => {} //~ ERROR this pattern has 2 fields, but the corresponding tuple variant has 3 fields
        E1::S(1, 2, 3, 4) => {} //~ ERROR this pattern has 4 fields, but the corresponding tuple variant has 3 fields
    }

    match E2::S(1, 2, 3) {
        E2::S() => {} //~ ERROR this pattern has 0 fields, but the corresponding tuple variant has 3 fields
        E2::S(1) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple variant has 3 fields
        E2::S(xyz, abc) => {} //~ ERROR this pattern has 2 fields, but the corresponding tuple variant has 3 fields
        E2::S(1, 2, 3, 4) => {} //~ ERROR this pattern has 4 fields, but the corresponding tuple variant has 3 fields
    }
    match E2::M(1, 2, 3) {
        E2::M() => {} //~ ERROR this pattern has 0 fields, but the corresponding tuple variant has 3 fields
        E2::M(1) => {} //~ ERROR this pattern has 1 field, but the corresponding tuple variant has 3 fields
        E2::M(xyz, abc) => {} //~ ERROR this pattern has 2 fields, but the corresponding tuple variant has 3 fields
        E2::M(1, 2, 3, 4) => {} //~ ERROR this pattern has 4 fields, but the corresponding tuple variant has 3 fields
    }
}
