# Guiding principles and rationale

When deciding on style guidelines, the style team follows these guiding
principles (in rough priority order):

* readability
    - scan-ability
    - avoiding misleading formatting
    - accessibility - readable and editable by users using the widest
      variety of hardware, including non-visual accessibility interfaces
    - readability of code in contexts without syntax highlighting or IDE
      assistance, such as rustc error messages, diffs, grep, and other
      plain-text contexts

* aesthetics
    - sense of 'beauty'
    - consistent with other languages/tools

* specifics
    - compatibility with version control practices - preserving diffs,
      merge-friendliness, etc.
    - preventing rightward drift
    - minimising vertical space

* application
    - ease of manual application
    - ease of implementation (in `rustfmt`, and in other tools/editors/code generators)
    - internal consistency
    - simplicity of formatting rules
