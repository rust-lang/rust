//@ rustfmt-unstable: true
//@ rustfmt-edition: 2024
//@ rustfmt-control_brace_style: AlwaysNextLine

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
