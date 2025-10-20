use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use crate::common::Config;
use crate::directives::{DirectiveLine, TestProps};

pub(crate) static DIRECTIVE_HANDLERS_MAP: LazyLock<HashMap<&str, Handler>> =
    LazyLock::new(make_directive_handlers_map);

#[derive(Clone)]
pub(crate) struct Handler {
    handler_fn: Arc<dyn Fn(HandlerArgs<'_>) + Send + Sync>,
}

impl Handler {
    pub(crate) fn handle(&self, config: &Config, line: &DirectiveLine<'_>, props: &mut TestProps) {
        (self.handler_fn)(HandlerArgs { config, line, props })
    }
}

struct HandlerArgs<'a> {
    config: &'a Config,
    line: &'a DirectiveLine<'a>,
    props: &'a mut TestProps,
}

/// Intermediate data structure, used for defining a list of handlers.
struct NamedHandler {
    names: Vec<&'static str>,
    handler: Handler,
}

/// Helper function to create a simple handler, so that changes can be made
/// to the handler struct without disturbing existing handler declarations.
fn handler(
    name: &'static str,
    handler_fn: impl Fn(&Config, &DirectiveLine<'_>, &mut TestProps) + Send + Sync + 'static,
) -> NamedHandler {
    multi_handler(&[name], handler_fn)
}

/// Associates the same handler function with multiple directive names.
fn multi_handler(
    names: &[&'static str],
    handler_fn: impl Fn(&Config, &DirectiveLine<'_>, &mut TestProps) + Send + Sync + 'static,
) -> NamedHandler {
    NamedHandler {
        names: names.to_owned(),
        handler: Handler {
            handler_fn: Arc::new(move |args| handler_fn(args.config, args.line, args.props)),
        },
    }
}

fn make_directive_handlers_map() -> HashMap<&'static str, Handler> {
    use crate::directives::directives::*;

    let handlers: Vec<NamedHandler> = vec![
        handler(ERROR_PATTERN, |config, ln, props| {
            config.push_name_value_directive(ln, ERROR_PATTERN, &mut props.error_patterns, |r| r);
        }),
        handler(REGEX_ERROR_PATTERN, |config, ln, props| {
            config.push_name_value_directive(
                ln,
                REGEX_ERROR_PATTERN,
                &mut props.regex_error_patterns,
                |r| r,
            );
        }),
    ];

    handlers
        .into_iter()
        .flat_map(|NamedHandler { names, handler }| {
            names.into_iter().map(move |name| (name, Handler::clone(&handler)))
        })
        .collect()
}
