cargo install --force

echo "Running integration test for crate ${INTEGRATION}"

git clone https://github.com/${INTEGRATION}.git

function check() {
  cargo clippy --all &> clippy_output
  cat clippy_output
  ! cat clippy_output | grep -q "internal error"
  if [[ $? != 0 ]]; then
    return 1
  fi
}

case ${INTEGRATION} in
  rust-lang/cargo)
    check
    ;;
  *)
    cd ${INTEGRATION}
    check
    ;;
esac
