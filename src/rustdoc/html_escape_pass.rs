#[doc = "Escapes characters that are not valid in HTML"];

export mk_pass;

fn mk_pass() -> pass {
    desc_pass::mk_pass("html_escape", escape)
}

fn escape(s: str) -> str {
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
