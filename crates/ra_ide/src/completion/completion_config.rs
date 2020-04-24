#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompletionConfig {
    pub enable_postfix_completions: bool,
    pub add_call_parenthesis: bool,
    pub add_call_argument_snippets: bool,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        CompletionConfig {
            enable_postfix_completions: true,
            add_call_parenthesis: true,
            add_call_argument_snippets: true,
        }
    }
}
