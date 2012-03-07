#[doc = "Escapes characters that are not valid in HTML"];

export mk_pass;

fn mk_pass() -> pass {
    desc_pass::mk_pass("html_escape", escape)
}

fn escape(s: str) -> str {
    let parts = str::split_char(s, '`');
    let i = 0;
    let parts = vec::map(parts) {|part|
        i += 1;
        if i % 2 != 0 {
            escape_(part)
        } else {
            part
        }
    };
    ret str::connect(parts, "`");
}

fn escape_(s: str) -> str {
    let s = str::replace(s, "&", "&amp;");
    let s = str::replace(s, "<", "&lt;");
    let s = str::replace(s, ">", "&gt;");
    let s = str::replace(s, "\"", "&quot;");
    ret s;
}

#[test]
fn test() {
    assert escape("<") == "&lt;";
    assert escape(">") == "&gt;";
    assert escape("&") == "&amp;";
    assert escape("\"") == "&quot;";
    assert escape("<>&\"") == "&lt;&gt;&amp;&quot;";
}

#[test]
fn should_not_escape_characters_in_backticks() {
    // Markdown will quote things in backticks itself
    assert escape("<`<`<`<`<") == "&lt;`<`&lt;`<`&lt;";
}
