// Issue 68293: This tests that the following changes work:
// the suggestion "format specifiers use curly braces: `{}`" is made
// found format specifiers are pointed at

fn no_format_specifiers_one_unused_argument() {
  println!("list: ", 1);
  //~^ ERROR argument never used
  //~| NOTE formatting specifier missing
  //~| NOTE format specifiers use curly braces: `{}`
  //~| NOTE argument never used
}

fn no_format_specifiers_multiple_unused_arguments() {
  println!("list: ", 3, 4, 5);
  //~^ ERROR multiple unused formatting arguments
  //~| NOTE multiple missing formatting specifiers
  //~| NOTE format specifiers use curly braces: `{}`
  //~| NOTE argument never used
  //~| NOTE argument never used
  //~| NOTE argument never used
}

fn missing_format_specifiers_one_unused_argument() {
  println!("list: a{}, b{}", 1, 2, 3);
  //~^ ERROR argument never used
  //~| NOTE formatting specifier missing
  //~| NOTE argument never used
}

fn missing_format_specifiers_multiple_unused_arguments() {
  println!("list: a{}, b{} c{}", 1, 2, 3, 4, 5);
  //~^ ERROR multiple unused formatting arguments
  //~| NOTE multiple missing formatting specifiers
  //~| NOTE argument never used
  //~| NOTE argument never used
}

fn main() {}
