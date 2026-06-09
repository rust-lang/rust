# Style for Templates

This directory has templates in the [Askama templating language][askamadoc], which is very
similar to [Jinja2][jinjadoc].

[jinjadoc]: https://jinja.palletsprojects.com/en/3.1.x/templates/
[askamadoc]: https://docs.rs/askama/latest/askama/

We want our rendered output to have as little unnecessary whitespace as
possible, so that pages load quickly. To achieve that we use Askama's
[whitespace control] features. By default, whitespace characters are removed
around jinja tags (`{% %}` for example). At the end of most lines, we put an
empty comment tag: `{# #}`. This causes all whitespace between the end of the
line and the beginning of the next, including indentation, to be omitted on
render. Sometimes we want to preserve a single space. In those cases we put the
space at the end of the line, followed by `{#+ #}`, which is a directive to
remove following whitespace but not preceding. We also use the whitespace
control characters in most instances of tags with control flow, for example
`{% if foo %}`.

[whitespace control]: https://askama.readthedocs.io/en/stable/configuration.html#whitespace-control

We want our templates to be readable, so we use indentation and newlines
liberally. We indent by four spaces after opening an HTML tag _or_ a Jinja
tag. In most cases an HTML tag should be followed by a newline, but if the
tag has simple contents and fits with its close tag on a single line, the
contents don't necessarily need a new line.

Askama templates support quite sophisticated control flow. To keep our templates
simple and understandable, we use only a subset: `if` and `for`. In particular
we avoid [assignments in the template logic][assignments] and [Askama
macros][macros]. This also may make things easier if we switch to a different
Jinja-style template system in the future.

[assignments]: https://askama.readthedocs.io/en/stable/template_syntax.html#assignments
[macros]: https://askama.readthedocs.io/en/stable/template_syntax.html#macros
