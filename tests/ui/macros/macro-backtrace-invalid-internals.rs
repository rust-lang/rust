// Macros in statement vs expression position handle backtraces differently.

macro_rules! fake_method_stmt {
     () => {
          1.fake() //~ ERROR no method
     }
}

macro_rules! fake_field_stmt {
     () => {
          1.fake //~ ERROR doesn't have fields
     }
}

macro_rules! fake_anon_field_stmt {
     () => {
          (1).0 //~ ERROR doesn't have fields
     }
}

macro_rules! fake_method_expr {
     () => {
          1.fake() //~ ERROR no method
     }
}

macro_rules! fake_field_expr {
     () => {
          1.fake //~ ERROR doesn't have fields
     }
}

macro_rules! fake_anon_field_expr {
     () => {
          (1).0 //~ ERROR doesn't have fields
     }
}

macro_rules! real_method_stmt {
     () => {
          2.0.neg() //~ ERROR can't call method `neg` on ambiguous numeric type `{float}`
     }
}

macro_rules! real_method_expr {
     () => {
          2.0.neg() //~ ERROR can't call method `neg` on ambiguous numeric type `{float}`
     }
}

fn main() {
    fake_method_stmt!();
    fake_field_stmt!();
    fake_anon_field_stmt!();
    real_method_stmt!();

    let _ = fake_method_expr!();
    let _ = fake_field_expr!();
    let _ = fake_anon_field_expr!();
    let _ = real_method_expr!();
}
