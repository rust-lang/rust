Formats the value using the given formatter.

# Errors

This function should return [`Err`] if, and only if, the provided [`Formatter`] returns [`Err`].
String formatting is considered an infallible operation; this function only
returns a [`Result`] because writing to the underlying stream might fail and it must
provide a way to propagate the fact that an error has occurred back up the stack.
