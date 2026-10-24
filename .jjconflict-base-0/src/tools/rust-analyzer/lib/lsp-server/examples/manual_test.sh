#!/usr/bin/env bash
# Simple nine-packet LSP test for examples/minimal_lsp.rs
# Usage (two tabs):
#
#   mkfifo /tmp/lsp_pipe          # one-time setup
#   # tab 1 – run the server
#   cat /tmp/lsp_pipe | cargo run --example minimal_lsp
#
#   # tab 2 – fire the packets (this script)
#   bash examples/manual_test.sh          # blocks until server exits
#
# If you don’t use a second tab, run the script in the background:
#
#   bash examples/manual_test.sh &        # writer in background
#   cat /tmp/lsp_pipe | cargo run --example minimal_lsp
#
# The script opens /tmp/lsp_pipe for writing (exec 3>) and sends each JSON
# packet with a correct Content-Length header.
#
# One-liner alternative (single terminal, no FIFO):
#
#   cargo run --example minimal_lsp <<'EOF'
#     … nine packets …
#   EOF
#
# Both approaches feed identical bytes to minimal_lsp via stdin.

set -eu
PIPE=${1:-/tmp/lsp_pipe}

mkfifo -m 600 "$PIPE" 2>/dev/null || true       # create once, ignore if exists

# open write end so the fifo stays open
exec 3> "$PIPE"

send() {
  local body=$1
  local len=$(printf '%s' "$body" | wc -c)
  printf 'Content-Length: %d\r\n\r\n%s' "$len" "$body" >&3
}

send '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}'
send '{"jsonrpc":"2.0","method":"initialized","params":{}}'
send '{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///tmp/foo.rs","languageId":"rust","version":1,"text":"fn  main( ){println!(\"hi\") }"}}}'
send '{"jsonrpc":"2.0","id":2,"method":"textDocument/completion","params":{"textDocument":{"uri":"file:///tmp/foo.rs"},"position":{"line":0,"character":0}}}'
send '{"jsonrpc":"2.0","id":3,"method":"textDocument/hover","params":{"textDocument":{"uri":"file:///tmp/foo.rs"},"position":{"line":0,"character":0}}}'
send '{"jsonrpc":"2.0","id":4,"method":"textDocument/definition","params":{"textDocument":{"uri":"file:///tmp/foo.rs"},"position":{"line":0,"character":0}}}'
send '{"jsonrpc":"2.0","id":5,"method":"textDocument/formatting","params":{"textDocument":{"uri":"file:///tmp/foo.rs"},"options":{"tabSize":4,"insertSpaces":true}}}'
send '{"jsonrpc":"2.0","id":6,"method":"shutdown","params":null}'
send '{"jsonrpc":"2.0","method":"exit","params":null}'

exec 3>&-
echo "Packets sent – watch the other terminal for responses."
