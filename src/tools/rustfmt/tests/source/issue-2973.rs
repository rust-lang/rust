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

  summary_test! {
    tokenize_strings,
    r#"a = "'a'" + '"b"' + "'c'" + '"d"'#echo hello"#,
    r#"N="+'+"+'#."#,
  }

  summary_test! {
        tokenize_recipe_interpolation_eol,
    "foo: # some comment
 {{hello}}
",
        "N:#$>^{N}$<.",
      }

  summary_test! {
        tokenize_recipe_interpolation_eof,
    "foo: # more comments
 {{hello}}
# another comment
",
        "N:#$>^{N}$<#$.",
      }

  summary_test! {
    tokenize_recipe_complex_interpolation_expression,
    "foo: #lol\n {{a + b + \"z\" + blarg}}",
    "N:#$>^{N+N+\"+N}<.",
  }

  summary_test! {
    tokenize_recipe_multiple_interpolations,
    "foo:,#ok\n {{a}}0{{b}}1{{c}}",
    "N:,#$>^{N}_{N}_{N}<.",
  }

  summary_test! {
        tokenize_junk,
    "bob

hello blah blah blah : a b c #whatever
    ",
        "N$$NNNN:NNN#$.",
      }

  summary_test! {
        tokenize_empty_lines,
    "
# this does something
hello:
  asdf
  bsdf

  csdf

  dsdf # whatever

# yolo
  ",
        "$#$N:$>^_$^_$$^_$$^_$$<#$.",
      }

  summary_test! {
        tokenize_comment_before_variable,
    "
#
A='1'
echo:
  echo {{A}}
  ",
        "$#$N='$N:$>^_{N}$<.",
      }

  summary_test! {
    tokenize_interpolation_backticks,
    "hello:\n echo {{`echo hello` + `echo goodbye`}}",
    "N:$>^_{`+`}<.",
  }

  summary_test! {
    tokenize_assignment_backticks,
    "a = `echo hello` + `echo goodbye`",
    "N=`+`.",
  }

  summary_test! {
        tokenize_multiple,
    "
hello:
  a
  b

  c

  d

# hello
bob:
  frank
  ",

        "$N:$>^_$^_$$^_$$^_$$<#$N:$>^_$<.",
      }

  summary_test! {
    tokenize_comment,
    "a:=#",
    "N:=#."
  }

  summary_test! {
    tokenize_comment_with_bang,
    "a:=#foo!",
    "N:=#."
  }

  summary_test! {
        tokenize_order,
    r"
b: a
  @mv a b

a:
  @touch F
  @touch a

d: c
  @rm c

c: b
  @mv b c",
        "$N:N$>^_$$<N:$>^_$^_$$<N:N$>^_$$<N:N$>^_<.",
      }

  summary_test! {
    tokenize_parens,
    r"((())) )abc(+",
    "((())))N(+.",
  }

  summary_test! {
    crlf_newline,
    "#\r\n#asdf\r\n",
    "#$#$.",
  }
}
