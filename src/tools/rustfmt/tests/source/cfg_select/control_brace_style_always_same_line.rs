//@ rustfmt-unstable: true
//@ rustfmt-control_brace_style: AlwaysSameLine
//                               ^ NB: default value

cfg_select! {
    _ => { x }
}

cfg_select! {
    _ => {
        x
    }
}

cfg_select! {
    _ => {{ x }}
}

cfg_select! {
    _ => {{
        x
    }}
}

cfg_select! {
    _ => unsafe { x }
}

cfg_select! {
    _ => {
        unsafe { x }
    }
}

cfg_select! {
    _ => // line comment
    {{ x }}
}

cfg_select! {
    _ => // line comment
    { a; b }
}

cfg_select! {
    _ => /* after */ {{ x }}
}

cfg_select! {
    _ => /* after */ {a; b}
}

cfg_select! {
    _ => /* after
    multiline */ {{ x }}
}

cfg_select! {
    _ => /* after
    multiline */ { a; b }
}

cfg_select! {
    _ => /* after */ {{
        x
    }}
}

cfg_select! {
    _ => /* after */ {
        a; b
    }
}

cfg_select! {
    _ /* this comment is _very_ long but brace should fit ---------------------------------- */ => { a; b }
}

cfg_select! {
    _ /* this comment is _very_ long and adds up to 101 columns incl. arrow ------------------- */ => { a; b }
}

cfg_select! {
    _ /* this comment is _very_ long and adds up to 100 columns itself -------------------------- */ => { a; b }
}

cfg_select! {
    _ /* this comment is _very_ long but shouldn't change next next line --------------------- */ =>
    {{ ThisLineIsShort }}
}

cfg_select! {
    _ /* this comment is _very_ long and adds up to 101 columns incl. arrow ------------------- */ =>
    {{ ThisLineIsShort }}
}

cfg_select! {
    _ /* this comment is _very_ long and adds up to 100 columns itself -------------------------- */ =>
    {{ ThisLineIsShort }}
}
