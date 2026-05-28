// rustfmt-newline_style: Windows
#[cfg(test)]
mod test {
    summary_test! {
        tokenize_recipe_interpolation_eol,
    "foo: # some comment
 {{hello}}
",
    "foo: \
 {{hello}} \
{{ahah}}",
        "N:#$>^{N}$<.",
      }
}
