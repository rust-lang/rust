#![register_tool(tool)] //~ ERROR the `register_tool` attribute is an experimental feature
#![register_attribute_tool(attr_tool)] //~ ERROR the `register_attribute_tool` attribute is an experimental feature
#![register_lint_tool(lint_tool)] //~ ERROR the `register_lint_tool` attribute is an experimental feature

fn main() {}
