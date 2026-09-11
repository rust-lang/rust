// rustfmt-style_edition: 2024
// rustfmt-format_macro_matchers: true

// From original issue example - Line length 101
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeio:expr, $add:expr) => {{
return;
}};
}

// Spaces between the `{` and `}`
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeio:expr, $add:expr) => {     {
return;
}      };
}

// Multi  `{}`
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeio:expr, $add:expr) => {{{{
return;
}}}};
}

// Multi  `{}` with spaces
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeio:expr, $add:expr) => {    {    {    {
return;
}     }     }    };
}
    
// Line length 102
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeiou:expr, $add:expr) => {{
return;
}};
}

// Line length 103
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeioua:expr, $add:expr) => {{
return;
}};
}

// With extended macro body - Line length 101
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeio:expr, $add:expr) => {{
let VAR = "VALUE"; return VAR;
}};
}

// With extended macro body - Line length 102
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeiou:expr, $add:expr) => {{
let VAR = "VALUE"; return VAR;
}};
}

// With extended macro body - Line length 103
macro_rules! test {
($aasdfghj:expr, $qwertyuiop:expr, $zxcvbnmasdfghjkl:expr, $aeiouaeiouaeioua:expr, $add:expr) => {{
let VAR = "VALUE"; return VAR;
}};
}
