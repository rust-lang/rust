extern "C" {
    void rust_panic();

    bool caught_panic() {
        try {
            rust_panic();
        } catch (...) {
            return true;
        }

        return false;
    }
}
