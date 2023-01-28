# `foo` isn't provided by this diagnostic so it is expected that the fallback message is used.
parse_struct_literal_body_without_path = this is a {$foo} message
    .suggestion = this is a test suggestion
