# `QNX_TARGET`

----

This environment variable is mandatory when linking on `nto-qnx*_iosock` platforms. It is used to determine an `-L` path to pass to the QNX linker.

You should [set this variable] by running `source qnxsdp-env.sh`.
See [the QNX docs] for more background information.

[set this variable]: https://www.qnx.com/developers/docs/qsc/com.qnx.doc.qsc.inst_larg_org/topic/build_server_developer_steps.html
[the QNX docs]: https://www.qnx.com/developers/docs/7.1/#com.qnx.doc.neutrino.io_sock/topic/migrate_app.html.
