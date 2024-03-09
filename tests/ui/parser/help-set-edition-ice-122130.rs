enum will {
    s#[c"owned_box"]
    //~^ERROR expected one of `(`, `,`, `=`, `{`, or `}`, found `#`
    //~|ERROR expected item, found `"owned_box"`
}
