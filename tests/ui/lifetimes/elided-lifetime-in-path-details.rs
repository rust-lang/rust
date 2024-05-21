//@revisions: tied untied

#![cfg_attr(tied, deny(tied_lifetimes_hidden_in_paths))]
#![cfg_attr(untied, deny(untied_lifetimes_hidden_in_paths))]

struct ContainsLifetime<'a>(&'a u8);

// ==========
// Core desired functionality

fn top_level_to_nested(v: &u8) ->
    ContainsLifetime
    //[tied]~^ ERROR hidden lifetime parameters
{
    ContainsLifetime(v)
}

fn nested_to_top_level(
    v: ContainsLifetime,
    //[tied]~^ ERROR hidden lifetime parameters
) -> &u8
{
    v.0
}

fn nested_to_nested(
    v: ContainsLifetime,
    //[tied]~^ ERROR hidden lifetime parameters
) -> ContainsLifetime
    //[tied]~^ ERROR hidden lifetime parameters
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

fn top_level_nested_parameter(v: &ContainsLifetime) {}
//[untied]~^ ERROR hidden lifetime parameters

// ==========
// More complicated types

fn top_level_to_multiple_nested(v: &u8) -> (
    ContainsLifetime,
    //[tied]~^ ERROR hidden lifetime parameters
    ContainsLifetime,
    //[tied]~^ ERROR hidden lifetime parameters
)
{
    (ContainsLifetime(v), ContainsLifetime(v))
}

// ----------

struct AsAMethod(u8);

impl AsAMethod {
    fn top_level_to_nested(v: &u8) ->
        ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
    {
        ContainsLifetime(v)
    }

    fn nested_to_top_level(
        v: ContainsLifetime,
        //[tied]~^ ERROR hidden lifetime parameters
    ) -> &u8
    {
        v.0
    }

    fn nested_to_nested(
        v: ContainsLifetime,
        //[tied]~^ ERROR hidden lifetime parameters
    ) -> ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
    {
        v
    }

    fn top_level_to_top_level(v: &u8) -> &u8 {
        v
    }

    fn self_to_nested(&self) ->
        ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
    {
        ContainsLifetime(&self.0)
    }

    fn self_to_nested_with_irrelevant_top_level_parameter(&self, _: &u8) ->
        ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
    {
        ContainsLifetime(&self.0)
    }

    fn self_to_nested_with_irrelevant_nested_parameter(
        &self,
        _: ContainsLifetime,
        //[untied]~^ ERROR hidden lifetime parameters
    ) -> ContainsLifetime
        //[tied]~^ ERROR hidden lifetime parameters
    {
        ContainsLifetime(&self.0)
    }

    fn nested_in_parameter(
        &self,
        v: ContainsLifetime,
        //[untied]~^ ERROR hidden lifetime parameters
    ) {}

    fn nested_in_parameter_with_return(
        &self,
        v: ContainsLifetime,
        //[untied]~^ ERROR hidden lifetime parameters
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
