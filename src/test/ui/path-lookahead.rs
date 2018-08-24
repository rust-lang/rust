// run-pass

#![warn(unused)]

// Parser test for #37765

fn with_parens<T: ToString>(arg: T) -> String { //~WARN function is never used: `with_parens`
  return (<T as ToString>::to_string(&arg)); //~WARN unnecessary parentheses around `return` value
}

fn no_parens<T: ToString>(arg: T) -> String { //~WARN function is never used: `no_parens`
  return <T as ToString>::to_string(&arg);
}

fn main() {

}
