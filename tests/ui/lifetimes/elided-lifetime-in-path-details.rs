//@revisions: tied untied

#![allow(elided_named_lifetimes)]

#![cfg_attr(tied, deny(elided_lifetimes_in_paths_tied))]
//[tied]~^ NOTE: the lint level is defined here
#![cfg_attr(untied, deny(elided_lifetimes_in_paths_untied))]
//[untied]~^ NOTE: the lint level is defined here

struct ContainsLifetime<'a>(&'a u8);

// ==========
// Core desired functionality

fn top_level_to_nested(v: &u8) ->
    //[tied]~^ NOTE lifetime comes from here
    ContainsLifetime
    //[tied]~^ ERROR hidden lifetime parameters
    //[tied]~| NOTE expected lifetime parameter
    //[tied]~| HELP indicate the anonymous lifetime
{
    ContainsLifetime(v)
}

fn nested_to_top_level(
    v: ContainsLifetime,
    //[tied]~^ ERROR hidden lifetime parameters
    //[tied]~| NOTE expected lifetime parameter
    //[tied]~| HELP indicate the anonymous lifetime
) -> &u8
{
    v.0
}

fn nested_to_nested(
    v: ContainsLifetime,
    //[tied]~^ ERROR hidden lifetime parameters
    //[tied]~| NOTE expected lifetime parameter
    //[tied]~| HELP indicate the anonymous lifetime
) -> ContainsLifetime
    //[tied]~^ NOTE expected lifetime parameter
    //[tied]~| HELP indicate the anonymous lifetime
{
    v
}

fn top_level_to_top_level(v: &u8) -> &u8 {
    v
}

// ==========
// Mixed named and elided lifetimes

fn named_top_level_to_nested<'a>(v: &'a u8) ->
    ContainsLifetime
    //[tied]~^ ERROR hidden lifetime parameters
    //[tied]~| NOTE expected lifetime parameter
    //[tied]~| HELP indicate the anonymous lifetime
{
    ContainsLifetime(v)
}

// ==========
// Using named lifetimes everywhere should not report

fn named_top_level_to_named_nested<'a>(v: &'a u8) -> ContainsLifetime<'a> {
    ContainsLifetime(v)
}

fn named_nested_to_named_top_level<'a>(v: ContainsLifetime<'a>) -> &'a u8 {
    v.0
}

fn named_nested_to_named_nested<'a>(v: ContainsLifetime<'a>) -> ContainsLifetime<'a> {
    v
}

// ==========
// Using anonymous lifetimes everywhere should not report

fn anon_top_level_to_anon_nested(v: &'_ u8) -> ContainsLifetime<'_> {
    ContainsLifetime(v)
}

fn anon_nested_to_anon_top_level(v: ContainsLifetime<'_>) -> &'_ u8 {
    v.0
}

fn anon_nested_to_anon_nested(v: ContainsLifetime<'_>) -> ContainsLifetime<'_> {
    v
}

// ==========
// Mixing named and anonymous lifetimes should not report

fn named_nested_to_anon_top_level<'a>(v: ContainsLifetime<'a>) -> &'_ u8 {
    v.0
}

fn named_top_level_to_anon_top_level<'a>(v: &'a u8) -> ContainsLifetime<'_> {
    ContainsLifetime(v)
}

// ==========
// Lifetimes with nothing to tie to

fn top_level_parameter(v: &u8) {}

fn nested_parameter(v: ContainsLifetime) {}
//[untied]~^ ERROR hidden lifetime parameters
//[untied]~| NOTE expected lifetime parameter

fn top_level_nested_parameter(v: &ContainsLifetime) {}
//[untied]~^ ERROR hidden lifetime parameters
//[untied]~| NOTE expected lifetime parameter

// ==========
// More complicated types

fn top_level_to_multiple_nested(v: &u8) -> (
    //[tied]~^ NOTE lifetime comes from here
    ContainsLifetime,
    //[tied]~^ ERROR hidden lifetime parameters
    //[tied]~| NOTE expected lifetime parameter
    //[tied]~| HELP indicate the anonymous lifetime
    ContainsLifetime,
    //[tied]~^ NOTE expected lifetime parameter
    //[tied]~| HELP indicate the anonymous lifetime
)
{
    (ContainsLifetime(v), ContainsLifetime(v))
}

// ----------

struct AsAMethod(u8);

impl AsAMethod {
    fn top_level_to_nested(
        v: &u8,
        //[tied]~^ NOTE lifetime comes from here
    ) ->
        ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
        //[tied]~| NOTE expected lifetime parameter
        //[tied]~| HELP indicate the anonymous lifetime
    {
        ContainsLifetime(v)
    }

    fn nested_to_top_level(
        v: ContainsLifetime,
        //[tied]~^ ERROR hidden lifetime parameters
        //[tied]~| NOTE expected lifetime parameter
        //[tied]~| HELP indicate the anonymous lifetime
    ) -> &u8
    {
        v.0
    }

    fn nested_to_nested(
        v: ContainsLifetime,
        //[tied]~^ ERROR hidden lifetime parameters
        //[tied]~| NOTE expected lifetime parameter
        //[tied]~| HELP indicate the anonymous lifetime
    ) -> ContainsLifetime
        //[tied]~^ NOTE expected lifetime parameter
        //[tied]~| HELP indicate the anonymous lifetime
    {
        v
    }

    fn top_level_to_top_level(v: &u8) -> &u8 {
        v
    }

    fn self_to_nested(
        &self,
        //[tied]~^ NOTE lifetime comes from here
    ) ->
        ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
        //[tied]~| NOTE expected lifetime parameter
        //[tied]~| HELP indicate the anonymous lifetime
    {
        ContainsLifetime(&self.0)
    }

    fn self_to_nested_with_irrelevant_top_level_parameter(
        &self,
        //[tied]~^ NOTE lifetime comes from here
        _: &u8
    ) ->
        ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
        //[tied]~| NOTE expected lifetime parameter
        //[tied]~| HELP indicate the anonymous lifetime
    {
        ContainsLifetime(&self.0)
    }

    fn self_to_nested_with_irrelevant_nested_parameter(
        &self,
        //[tied]~^ NOTE lifetime comes from here
        _: ContainsLifetime,
        //[untied]~^ ERROR hidden lifetime parameters
        //[untied]~| NOTE expected lifetime parameter
    ) -> ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
        //[tied]~| NOTE expected lifetime parameter
        //[tied]~| HELP indicate the anonymous lifetime
    {
        ContainsLifetime(&self.0)
    }

    fn nested_in_parameter(
        &self,
        v: ContainsLifetime,
        //[untied]~^ ERROR hidden lifetime parameters
        //[untied]~| NOTE expected lifetime parameter
    ) {}

    fn nested_in_parameter_with_return(
        &self,
        v: ContainsLifetime,
        //[untied]~^ ERROR hidden lifetime parameters
        //[untied]~| NOTE expected lifetime parameter
    ) -> &u8
    {
        &self.0
    }
}

// // Do we need to worry about nested function signatures?
// // fn outer(_: fn(&) -> &)

// // Do we need to worry about closures?

// // Do we need to write tests for `self: Foo` syntax?

fn main() {}
