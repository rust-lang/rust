//@ check-pass
#![warn(unused_parens)]

fn id<T>(t: T) -> T { t }

fn main() {
    // This should not warn
    let _ = 1 as (i32) < 2;
    let _ = id(1 as (i32) < 2);
    let _ = id(1 as i32)
        as (i32) < 2;
    // These should warn
    let _ = 1 as ((i32)) < 2; //~WARN unnecessary parentheses
    let _ = 1 as (i32); //~WARN unnecessary parentheses
    let _ = 1 as (i32) > 2; //~WARN unnecessary parentheses
    let _ = id(1 as (i32)) //~WARN unnecessary parentheses
        as (i32) < 2;
    let _ = id(1 as (i32)) < 2; //~WARN unnecessary parentheses

    // This should not warn
    let _ = 1 as (i32) << 2;
    let _ = id(1 as (i32) << 2);
    let _ = id(1 as i32)
        as (i32) << 2;
    // These should warn
    let _ = 1 as ((i32)) << 2; //~WARN unnecessary parentheses
    let _ = 1 as (i32); //~WARN unnecessary parentheses
    let _ = 1 as (i32) >> 2; //~WARN unnecessary parentheses
    let _ = id(1 as (i32)) //~WARN unnecessary parentheses
        as (i32) << 2;
    let _ = id(1 as (i32)) << 2; //~WARN unnecessary parentheses
}
